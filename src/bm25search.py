"""
BM25Search - Clean BM25-based search engine.

This package provides a production-ready BM25 search engine with:
- Pure BM25 ranking algorithm
- Multilingual support (Russian and English)
- Persistent storage using SQLite
- Stemming and lemmatization
- Stop word filtering

Example:
    >>> import bm25search
    >>> 
    >>> # Quick search
    >>> results = bm25search.smart_search(["Hello world", "Python programming"], "programming")
    >>> 
    >>> # Advanced usage
    >>> engine = bm25search.SmartSearchEngine()
    >>> engine.index_documents(["Document 1", "Document 2"])
    >>> results = engine.search("query", top_k=10)
"""

import math
import re
import sqlite3
import json
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import warnings
import inspect

warnings.filterwarnings('ignore')

# Compatibility shim: some third-party libs still expect inspect.getargspec (returns 4-tuple)
if not hasattr(inspect, 'getargspec'):
    def _compat_getargspec(func):  # type: ignore
        fas = inspect.getfullargspec(func)  # returns 7-tuple
        return (fas.args, fas.varargs, fas.varkw, fas.defaults)
    try:
        inspect.getargspec = _compat_getargspec  # type: ignore[attr-defined]
    except Exception:
        pass

try:
    import nltk
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import pymorphy2
    PYMORPHY2_AVAILABLE = True
except ImportError:
    PYMORPHY2_AVAILABLE = False

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False


class SmartSearchEngine:
    """
    Clean BM25 search engine with stemming and lemmatization support.
    
    Features:
    - BM25 ranking algorithm
    - Persistent storage using SQLite
    - Multilingual support (Russian and English)
    - Stemming and lemmatization
    - Stop word filtering
    
    Example:
        >>> engine = SmartSearchEngine()
        >>> engine.index_documents(["The cat sits on the mat", "Dogs are loyal animals"])
        >>> results = engine.search("cat", top_k=5)
        >>> print(results[0])
        (0, 2.45, "The cat sits on the mat")
    """
    
    def __init__(self, 
                 db_path: str = "search_index.db",
                 k1: float = 1.5, 
                 b: float = 0.75,
                 original_weight: float = 1.0,
                 normalized_weight: float = 0.6,
                 exact_surface_bonus: float = 0.5,
                 use_bigrams: bool = True,
                 use_trigrams: bool = False,
                 phrase_weight: float = 1.3,
                 trigram_weight: float = 1.6,
                 min_ngram_df: int = 1,
                 proximity_bonus: float = 0.3,
                 proximity_window: int = 3):
        """
        Initialize the search engine.
        
        Args:
            db_path: Path to SQLite database file
            k1: BM25 term frequency saturation parameter (1.2-2.0)
            b: BM25 document length normalization parameter (0.0-1.0)
        """
        self.db_path = db_path
        self.k1 = k1
        self.b = b
        self.conn = None
        self.original_weight = original_weight
        self.normalized_weight = normalized_weight
        self._setup_database()
        self._setup_language_tools()
        self.exact_surface_bonus = exact_surface_bonus
        self.use_bigrams = use_bigrams
        self.use_trigrams = use_trigrams
        self.phrase_weight = phrase_weight
        self.trigram_weight = trigram_weight
        self.min_ngram_df = min_ngram_df
        self.proximity_bonus = proximity_bonus
        self.proximity_window = proximity_window
        
    def _setup_database(self):
        """Initialize SQLite database with required tables."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                language TEXT,
                processed_terms TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS term_index (
                term TEXT,
                doc_id INTEGER,
                tf INTEGER,
                positions TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (id)
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS idf_scores (
                term TEXT PRIMARY KEY,
                idf REAL,
                df INTEGER
            )
        ''')
        
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_term ON term_index (term)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON term_index (doc_id)')
        self.conn.commit()
    
    def _setup_language_tools(self):
        """Initialize text processing tools for different languages."""
        self.processors = {}

        # Local capability detection to avoid stale globals
        nltk_available = False
        nl_stemmer_cls = None
        nl_stopwords = None
        try:
            import nltk  # type: ignore
            from nltk.stem import SnowballStemmer as _SB  # type: ignore
            from nltk.corpus import stopwords as _SW  # type: ignore
            nltk_available = True
            nl_stemmer_cls = _SB
            nl_stopwords = _SW
            # Ensure required resources are present
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except Exception:
            pass

        morph_analyzer = None
        self._lemma_init_error = None
        try:
            import pymorphy2 as _pm  # type: ignore
            morph_analyzer = _pm.MorphAnalyzer()
        except Exception as e:
            # Keep the error message for diagnostics
            self._lemma_init_error = str(e)
            morph_analyzer = None

        # Base stopwords fallbacks
        russian_stopwords = {
            'и', 'в', 'на', 'с', 'по', 'для', 'не', 'от', 'до', 'из', 'к', 'о', 'что', 'как', 'это',
            'а', 'но', 'или', 'да', 'нет', 'то', 'так', 'уже', 'еще', 'все', 'был', 'была', 'было',
            'были', 'есть', 'будет', 'будут', 'может', 'можно', 'нужно', 'надо', 'очень', 'более',
            'самый', 'такой', 'этот', 'тот', 'который', 'где', 'когда', 'почему', 'зачем', 'как'
        }
        english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us'
        }

        # Extend stopwords from NLTK if available
        if nltk_available and nl_stopwords is not None:
            try:
                russian_stopwords.update(set(nl_stopwords.words('russian')))
            except Exception:
                pass
            try:
                english_stopwords.update(set(nl_stopwords.words('english')))
            except Exception:
                pass

        # Compose RU processor
        ru_stemmer = None
        if nltk_available and nl_stemmer_cls is not None:
            try:
                ru_stemmer = nl_stemmer_cls('russian')
            except Exception:
                ru_stemmer = None
        self.processors['ru'] = {
            'stemmer': ru_stemmer,
            'lemmatizer': morph_analyzer,
            'stopwords': russian_stopwords,
        }

        # Compose EN processor
        en_stemmer = None
        if nltk_available and nl_stemmer_cls is not None:
            try:
                en_stemmer = nl_stemmer_cls('english')
            except Exception:
                en_stemmer = None
        self.processors['en'] = {
            'stemmer': en_stemmer,
            'lemmatizer': None,
            'stopwords': english_stopwords,
        }
    
    def _load_fasttext_model(self):
        """Deprecated: fastText disabled. Always returns None."""
        return None
    
    def _detect_language(self, text: str) -> str:
        """Detect language: langid (primary) -> langdetect -> heuristic."""
        txt = (text or '').strip()
        if not txt:
            return 'en'
        # 1) langid
        if 'LANGID_AVAILABLE' in globals() and LANGID_AVAILABLE:
            try:
                code, score = langid.classify(txt)
                return 'ru' if code == 'ru' else 'en'
            except Exception:
                pass
        # 2) langdetect fallback
        if LANGDETECT_AVAILABLE:
            try:
                lang = detect(txt)
                return 'ru' if lang == 'ru' else 'en'
            except Exception:
                pass
        # 3) simple heuristic
        if any(ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for ch in txt.lower()):
            return 'ru'
        return 'en'
    
    def _preprocess_text(self, text: str, language: str = None) -> Tuple[List[str], List[int]]:
        """
        Preprocess text with advanced features.
        
        Returns:
            Tuple of (processed_terms, positions)
        """
        if not text:
            return [], []
        
        if language is None:
            language = self._detect_language(text)
        
        # Basic cleaning with position tracking
        text = text.lower()
        words = []
        positions = []
        
        for match in re.finditer(r'\b\w+\b', text):
            word = match.group()
            if len(word) > 2:
                words.append(word)
                positions.append(match.start())
        
        # Get processor for language
        processor = self.processors.get(language, {})
        # Lazy retry to initialize RU lemmatizer if missing
        if language == 'ru' and not processor.get('lemmatizer'):
            try:
                import pymorphy2 as _pm  # type: ignore
                processor['lemmatizer'] = _pm.MorphAnalyzer()
                # persist in processors map
                self.processors['ru'] = processor
            except Exception as e:
                self._lemma_init_error = str(e)
        stopwords_set = processor.get('stopwords', set())
        
        # Filter stopwords and apply stemming/lemmatization
        filtered_words = []
        filtered_positions = []
        
        for word, pos in zip(words, positions):
            if word not in stopwords_set:
                # Always add original token
                filtered_words.append(word)
                filtered_positions.append(pos)

                # Language-specific normalization
                if language == 'ru':
                    # Russian: prefer lemmatization; fallback to stemming if lemmatizer unavailable
                    if processor.get('lemmatizer'):
                        try:
                            lemma = processor['lemmatizer'].parse(word)[0].normal_form
                            if len(lemma) > 2 and lemma != word and lemma not in filtered_words:
                                filtered_words.append(lemma)
                                filtered_positions.append(pos)
                        except:
                            pass
                    elif processor.get('stemmer'):
                        try:
                            stem = processor['stemmer'].stem(word)
                            if len(stem) > 2 and stem != word and stem not in filtered_words:
                                filtered_words.append(stem)
                                filtered_positions.append(pos)
                        except:
                            pass
                elif language == 'en':
                    # English: prefer stemming only
                    if processor.get('stemmer'):
                        try:
                            stem = processor['stemmer'].stem(word)
                            if len(stem) > 2 and stem != word and stem not in filtered_words:
                                filtered_words.append(stem)
                                filtered_positions.append(pos)
                        except:
                            pass
                else:
                    # Fallback for other languages: try stem first, else lemma
                    if processor.get('stemmer'):
                        try:
                            stem = processor['stemmer'].stem(word)
                            if len(stem) > 2 and stem != word and stem not in filtered_words:
                                filtered_words.append(stem)
                                filtered_positions.append(pos)
                        except:
                            pass
                    elif processor.get('lemmatizer'):
                        try:
                            lemma = processor['lemmatizer'].parse(word)[0].normal_form
                            if len(lemma) > 2 and lemma != word and lemma not in filtered_words:
                                filtered_words.append(lemma)
                                filtered_positions.append(pos)
                        except:
                            pass
        
        return filtered_words, filtered_positions

    def _extract_original_terms(self, text: str, language: str = None) -> Set[str]:
        """Tokenize and filter stopwords, but do NOT add stems/lemmas. Used for weighting."""
        if not text:
            return set()
        if language is None:
            language = self._detect_language(text)
        processor = self.processors.get(language, {})
        stopwords_set = processor.get('stopwords', set())
        terms = set()
        for match in re.finditer(r'\b\w+\b', text):
            word = match.group()
            if len(word) > 2 and word not in stopwords_set:
                terms.add(word)
        return terms

    def _tokenize_original_sequence(self, text: str, language: str = None) -> Tuple[List[str], List[int]]:
        """Return ordered sequence of original tokens (no stemming/lemmatization), positions are character offsets."""
        if not text:
            return [], []
        if language is None:
            language = self._detect_language(text)
        processor = self.processors.get(language, {})
        stopwords_set = processor.get('stopwords', set())
        words: List[str] = []
        positions: List[int] = []
        for match in re.finditer(r'\b\w+\b', text):
            word = match.group()
            if len(word) > 2 and word not in stopwords_set:
                words.append(word)
                positions.append(match.start())
        return words, positions
    
    def index_documents(self, documents: List[str]) -> None:
        """
        Index a collection of documents.
        
        Args:
            documents: List of document strings to index
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # Clear existing data
        self.conn.execute('DELETE FROM documents')
        self.conn.execute('DELETE FROM term_index')
        self.conn.execute('DELETE FROM idf_scores')
        
        # Index each document
        term_frequencies = defaultdict(int)
        
        for doc_id, doc_content in enumerate(documents):
            if not isinstance(doc_content, str):
                raise TypeError(f"Document {doc_id} must be a string")
            
            language = self._detect_language(doc_content)
            terms, positions = self._preprocess_text(doc_content, language)
            orig_words_seq, orig_pos_seq = self._tokenize_original_sequence(doc_content, language)
            
            # Save document
            self.conn.execute('''
                INSERT INTO documents (id, content, language, processed_terms)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, doc_content, language, json.dumps(terms)))
            
            # Index terms with positions
            term_positions = defaultdict(list)
            for term, pos in zip(terms, positions):
                term_positions[term].append(pos)
            
            for term, pos_list in term_positions.items():
                tf = len(pos_list)
                term_frequencies[term] += 1
                
                self.conn.execute('''
                    INSERT INTO term_index (term, doc_id, tf, positions)
                    VALUES (?, ?, ?, ?)
                ''', (term, doc_id, tf, json.dumps(pos_list)))

            # Index bigrams from original token sequence
            if self.use_bigrams and len(orig_words_seq) >= 2:
                bigram_positions = defaultdict(list)
                for i in range(len(orig_words_seq) - 1):
                    bg = f"{orig_words_seq[i]} {orig_words_seq[i+1]}"
                    # use start position of the first token
                    bigram_positions[bg].append(orig_pos_seq[i])
                for bg, pos_list in bigram_positions.items():
                    tf = len(pos_list)
                    term_frequencies[bg] += 1
                    self.conn.execute('''
                        INSERT INTO term_index (term, doc_id, tf, positions)
                        VALUES (?, ?, ?, ?)
                    ''', (bg, doc_id, tf, json.dumps(pos_list)))

            # Index trigrams from original token sequence
            if self.use_trigrams and len(orig_words_seq) >= 3:
                trigram_positions = defaultdict(list)
                for i in range(len(orig_words_seq) - 2):
                    tg = f"{orig_words_seq[i]} {orig_words_seq[i+1]} {orig_words_seq[i+2]}"
                    trigram_positions[tg].append(orig_pos_seq[i])
                for tg, pos_list in trigram_positions.items():
                    tf = len(pos_list)
                    term_frequencies[tg] += 1
                    self.conn.execute('''
                        INSERT INTO term_index (term, doc_id, tf, positions)
                        VALUES (?, ?, ?, ?)
                    ''', (tg, doc_id, tf, json.dumps(pos_list)))
        
        # Calculate and save IDF scores
        total_docs = len(documents)
        for term, df in term_frequencies.items():
            # Apply df threshold for n-grams (>1 token)
            if (' ' in term) and df < self.min_ngram_df:
                continue
            idf = math.log((total_docs - df + 0.5) / (df + 0.5))
            self.conn.execute('''
                INSERT INTO idf_scores (term, idf, df)
                VALUES (?, ?, ?)
            ''', (term, idf, df))
        
        self.conn.commit()
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of tuples (doc_id, score, content) sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        language = self._detect_language(query)
        query_terms, query_positions = self._preprocess_text(query, language)
        original_query_terms = self._extract_original_terms(query, language)
        
        if not query_terms:
            return []
        
        # Get candidate documents
        candidate_docs = set()
        for term in query_terms:
            cursor = self.conn.execute(
                'SELECT doc_id FROM term_index WHERE term = ?', (term,)
            )
            candidate_docs.update(row[0] for row in cursor.fetchall())

        # Include candidates by bigrams
        query_bigrams: List[str] = []
        if self.use_bigrams:
            orig_seq, _ = self._tokenize_original_sequence(query, language)
            for i in range(len(orig_seq) - 1):
                bg = f"{orig_seq[i]} {orig_seq[i+1]}"
                query_bigrams.append(bg)
                cursor = self.conn.execute(
                    'SELECT doc_id FROM term_index WHERE term = ?', (bg,)
                )
                candidate_docs.update(row[0] for row in cursor.fetchall())
        # Include candidates by trigrams
        query_trigrams: List[str] = []
        if self.use_trigrams:
            orig_seq, _ = self._tokenize_original_sequence(query, language)
            for i in range(len(orig_seq) - 2):
                tg = f"{orig_seq[i]} {orig_seq[i+1]} {orig_seq[i+2]}"
                query_trigrams.append(tg)
                cursor = self.conn.execute(
                    'SELECT doc_id FROM term_index WHERE term = ?', (tg,)
                )
                candidate_docs.update(row[0] for row in cursor.fetchall())
        
        if not candidate_docs:
            return []
        
        # Get average document length (mean token count)
        cursor = self.conn.execute('SELECT processed_terms FROM documents')
        rows = cursor.fetchall()
        lengths = [len(json.loads(r[0])) for r in rows]
        avg_doc_length = (sum(lengths) / len(lengths)) if lengths else 1
        
        # Calculate BM25 scores
        results = []
        for doc_id in candidate_docs:
            bm25_score = 0.0
            
            # Get document length
            cursor = self.conn.execute(
                'SELECT processed_terms FROM documents WHERE id = ?', (doc_id,)
            )
            doc_terms = json.loads(cursor.fetchone()[0])
            doc_length = len(doc_terms)
            
            for term in query_terms:
                # BM25 component
                cursor = self.conn.execute(
                    'SELECT tf FROM term_index WHERE term = ? AND doc_id = ?', 
                    (term, doc_id)
                )
                tf_result = cursor.fetchone()
                if tf_result:
                    tf = tf_result[0]
                    
                    cursor = self.conn.execute(
                        'SELECT idf FROM idf_scores WHERE term = ?', (term,)
                    )
                    idf_result = cursor.fetchone()
                    if idf_result:
                        idf = idf_result[0]
                        
                        # BM25 formula
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                        weight = self.original_weight if term in original_query_terms else self.normalized_weight
                        bm25_score += weight * idf * (numerator / denominator)

            # Add phrase (bigram) scores
            if self.use_bigrams and query_bigrams:
                for bg in query_bigrams:
                    cursor = self.conn.execute(
                        'SELECT tf FROM term_index WHERE term = ? AND doc_id = ?',
                        (bg, doc_id)
                    )
                    tf_result = cursor.fetchone()
                    if tf_result:
                        tf = tf_result[0]
                        cursor = self.conn.execute(
                            'SELECT idf FROM idf_scores WHERE term = ?', (bg,)
                        )
                        idf_result = cursor.fetchone()
                        if idf_result:
                            idf = idf_result[0]
                            numerator = tf * (self.k1 + 1)
                            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                            bm25_score += self.phrase_weight * idf * (numerator / denominator)

            # Add phrase (trigram) scores
            if self.use_trigrams and query_trigrams:
                for tg in query_trigrams:
                    cursor = self.conn.execute(
                        'SELECT tf FROM term_index WHERE term = ? AND doc_id = ?',
                        (tg, doc_id)
                    )
                    tf_result = cursor.fetchone()
                    if tf_result:
                        tf = tf_result[0]
                        cursor = self.conn.execute(
                            'SELECT idf FROM idf_scores WHERE term = ?', (tg,)
                        )
                        idf_result = cursor.fetchone()
                        if idf_result:
                            idf = idf_result[0]
                            numerator = tf * (self.k1 + 1)
                            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                            bm25_score += self.trigram_weight * idf * (numerator / denominator)

            cursor = self.conn.execute(
                'SELECT content FROM documents WHERE id = ?', (doc_id,)
            )
            content_row = cursor.fetchone()
            if content_row is not None:
                content = content_row[0]
                # Exact surface-form bonus: add per original token present as a whole word
                bonus = 0.0
                content_lower = content.lower()
                for tok in original_query_terms:
                    if re.search(r'\b' + re.escape(tok.lower()) + r'\b', content_lower):
                        bonus += self.exact_surface_bonus
                # Proximity bonus: reward close occurrences of different original tokens
                if self.proximity_bonus and len(original_query_terms) >= 2:
                    orig_list = list(original_query_terms)
                    window_chars = max(1, self.proximity_window * 10)
                    for i in range(len(orig_list)):
                        for j in range(i+1, len(orig_list)):
                            t1, t2 = orig_list[i], orig_list[j]
                            # get positions lists
                            p1 = self.conn.execute('SELECT positions FROM term_index WHERE term = ? AND doc_id = ?', (t1, doc_id)).fetchone()
                            p2 = self.conn.execute('SELECT positions FROM term_index WHERE term = ? AND doc_id = ?', (t2, doc_id)).fetchone()
                            if p1 and p2:
                                pos1 = json.loads(p1[0])
                                pos2 = json.loads(p2[0])
                                # minimal absolute char distance
                                mind = None
                                for a in pos1:
                                    for b in pos2:
                                        d = abs(a - b)
                                        if mind is None or d < mind:
                                            mind = d
                                if mind is not None and mind <= window_chars:
                                    bonus += self.proximity_bonus * (window_chars - mind) / window_chars
                results.append((doc_id, bm25_score + bonus, content))
        
        # Sort and return results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def explain_search(self, query: str, doc_id: int) -> Dict:
        """
        Explain how the search score was calculated for a specific document.
        
        Args:
            query: Search query
            doc_id: Document ID to explain
            
        Returns:
            Dictionary with detailed scoring information
        """
        language = self._detect_language(query)
        query_terms, query_positions = self._preprocess_text(query, language)
        original_query_terms = self._extract_original_terms(query, language)
        query_bigrams: List[str] = []
        if self.use_bigrams:
            orig_seq, _ = self._tokenize_original_sequence(query, language)
            for i in range(len(orig_seq) - 1):
                query_bigrams.append(f"{orig_seq[i]} {orig_seq[i+1]}")
        # Build trigrams for explain
        query_trigrams: List[str] = []
        if self.use_trigrams:
            orig_seq, _ = self._tokenize_original_sequence(query, language)
            for i in range(len(orig_seq) - 2):
                query_trigrams.append(f"{orig_seq[i]} {orig_seq[i+1]} {orig_seq[i+2]}")
        
        # Get document
        cursor = self.conn.execute(
            'SELECT content, language FROM documents WHERE id = ?', (doc_id,)
        )
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Document {doc_id} not found")
        
        doc_content, doc_language = result
        
        # Calculate score components
        bm25_score = 0.0
        term_details = {}
        
        cursor = self.conn.execute(
            'SELECT processed_terms FROM documents WHERE id = ?', (doc_id,)
        )
        doc_terms = json.loads(cursor.fetchone()[0])
        doc_length = len(doc_terms)
        
        cursor = self.conn.execute('SELECT processed_terms FROM documents')
        rows = cursor.fetchall()
        lengths = [len(json.loads(r[0])) for r in rows]
        avg_doc_length = (sum(lengths) / len(lengths)) if lengths else 1
        
        for term in query_terms:
            cursor = self.conn.execute(
                'SELECT tf FROM term_index WHERE term = ? AND doc_id = ?', 
                (term, doc_id)
            )
            tf_result = cursor.fetchone()
            
            if tf_result:
                tf = tf_result[0]
                
                cursor = self.conn.execute(
                    'SELECT idf FROM idf_scores WHERE term = ?', (term,)
                )
                idf = cursor.fetchone()[0]
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                weight = self.original_weight if term in original_query_terms else self.normalized_weight
                term_score = weight * idf * (numerator / denominator)
                bm25_score += term_score
                
                term_details[term] = {
                    'tf': tf,
                    'idf': idf,
                    'weight': weight,
                    'is_original': term in original_query_terms,
                    'score': term_score
                }
        # Phrase (bigram) components
        if self.use_bigrams and query_bigrams:
            for bg in query_bigrams:
                cursor = self.conn.execute(
                    'SELECT tf FROM term_index WHERE term = ? AND doc_id = ?', 
                    (bg, doc_id)
                )
                tf_result = cursor.fetchone()
                if tf_result:
                    tf = tf_result[0]
                    cursor = self.conn.execute('SELECT idf FROM idf_scores WHERE term = ?', (bg,))
                    idf_row = cursor.fetchone()
                    if idf_row:
                        idf = idf_row[0]
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                        term_score = self.phrase_weight * idf * (numerator / denominator)
                        bm25_score += term_score
                        term_details[bg] = {
                            'tf': tf,
                            'idf': idf,
                            'weight': self.phrase_weight,
                            'is_original': True,
                            'score': term_score,
                            'type': 'bigram'
                        }
        # Phrase (trigram) components
        if self.use_trigrams and query_trigrams:
            for tg in query_trigrams:
                cursor = self.conn.execute(
                    'SELECT tf FROM term_index WHERE term = ? AND doc_id = ?', 
                    (tg, doc_id)
                )
                tf_result = cursor.fetchone()
                if tf_result:
                    tf = tf_result[0]
                    cursor = self.conn.execute('SELECT idf FROM idf_scores WHERE term = ?', (tg,))
                    idf_row = cursor.fetchone()
                    if idf_row:
                        idf = idf_row[0]
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
                        term_score = self.trigram_weight * idf * (numerator / denominator)
                        bm25_score += term_score
                        term_details[tg] = {
                            'tf': tf,
                            'idf': idf,
                            'weight': self.trigram_weight,
                            'is_original': True,
                            'score': term_score,
                            'type': 'trigram'
                        }
        # Exact surface bonus details
        exact_matches = []
        bonus = 0.0
        content_lower = doc_content.lower()
        for tok in original_query_terms:
            if re.search(r'\b' + re.escape(tok.lower()) + r'\b', content_lower):
                exact_matches.append(tok)
                bonus += self.exact_surface_bonus
        # Proximity bonus details
        proximity_pairs = []
        if self.proximity_bonus and len(original_query_terms) >= 2:
            orig_list = list(original_query_terms)
            window_chars = max(1, self.proximity_window * 10)
            for i in range(len(orig_list)):
                for j in range(i+1, len(orig_list)):
                    t1, t2 = orig_list[i], orig_list[j]
                    p1 = self.conn.execute('SELECT positions FROM term_index WHERE term = ? AND doc_id = ?', (t1, doc_id)).fetchone()
                    p2 = self.conn.execute('SELECT positions FROM term_index WHERE term = ? AND doc_id = ?', (t2, doc_id)).fetchone()
                    if p1 and p2:
                        pos1 = json.loads(p1[0])
                        pos2 = json.loads(p2[0])
                        mind = None
                        for a in pos1:
                            for b in pos2:
                                d = abs(a - b)
                                if mind is None or d < mind:
                                    mind = d
                        if mind is not None and mind <= window_chars:
                            inc = self.proximity_bonus * (window_chars - mind) / window_chars
                            bonus += inc
                            proximity_pairs.append({'pair': (t1, t2), 'min_distance_chars': mind, 'bonus': inc})
        bm25_score += bonus
        
        return {
            'query': query,
            'document': doc_content,
            'query_language': language,
            'document_language': doc_language,
            'processed_query_terms': query_terms,
            'bm25_score': bm25_score,
            'final_score': bm25_score,
            'term_details': term_details,
            'exact_surface_matches': exact_matches,
            'exact_surface_bonus': bonus,
            'proximity_pairs': proximity_pairs,
            'use_bigrams': self.use_bigrams
        }
    
    def get_stats(self) -> Dict:
        """Get search engine statistics."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT COUNT(DISTINCT term) FROM term_index')
        total_terms = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT language, COUNT(*) FROM documents GROUP BY language')
        languages = dict(cursor.fetchall())
        
        return {
            'total_documents': total_docs,
            'total_unique_terms': total_terms,
            'languages': languages,
            'features': {
                'persistent_storage': True,
                'multilingual_support': True,
                'stemming': NLTK_AVAILABLE,
                'lemmatization': PYMORPHY2_AVAILABLE
            },
            'language_detection': LANGDETECT_AVAILABLE,
            'database_path': self.db_path,
            'lemma_error': getattr(self, '_lemma_init_error', None)
        }
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def smart_search(documents: List[str], query: str, top_k: int = 10, **kwargs) -> List[Tuple[int, float, str]]:
    """
    Quick search function without creating a persistent engine.
    
    Args:
        documents: List of documents to search
        query: Search query
        top_k: Maximum results to return
        **kwargs: Additional arguments for SmartSearchEngine
        
    Returns:
        List of (doc_id, score, content) tuples
    """
    engine = SmartSearchEngine(**kwargs)
    try:
        engine.index_documents(documents)
        return engine.search(query, top_k)
    finally:
        engine.close()


def batch_search(documents: List[str], queries: List[str], top_k: int = 10, **kwargs) -> Dict[str, List[Tuple[int, float, str]]]:
    """
    Perform multiple searches on the same document set.
    
    Args:
        documents: List of documents to search
        queries: List of search queries
        top_k: Maximum results per query
        **kwargs: Additional arguments for SmartSearchEngine
        
    Returns:
        Dictionary mapping queries to their results
    """
    engine = SmartSearchEngine(**kwargs)
    try:
        engine.index_documents(documents)
        results = {}
        for query in queries:
            results[query] = engine.search(query, top_k)
        return results
    finally:
        engine.close()


__version__ = "1.0.0"
__author__ = "BM25Search Team"
__email__ = "contact@bm25search.com"

__all__ = ['SmartSearchEngine', 'smart_search', 'batch_search']

# Convenience imports for backward compatibility
BM25Search = SmartSearchEngine
quick_search = smart_search
