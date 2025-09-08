"""
Basic tests for BM25Search functionality.
"""

import pytest
import tempfile
import os
from bm25search import SmartSearchEngine, smart_search


class TestSmartSearch:
    """Test suite for smart_search function."""
    
    def test_basic_search(self):
        """Test basic search functionality."""
        documents = [
            "The cat sits on the mat",
            "Dogs are loyal animals",
            "Cats and dogs are pets"
        ]
        
        results = smart_search("cat", documents, top_k=2)
        
        assert len(results) <= 2
        assert len(results) > 0
        
        # Check result format
        doc_id, score, content = results[0]
        assert isinstance(doc_id, int)
        assert isinstance(score, float)
        assert isinstance(content, str)
        assert score > 0
    
    def test_empty_query(self):
        """Test empty query handling."""
        documents = ["test document"]
        results = smart_search("", documents)
        assert len(results) == 0
    
    def test_empty_documents(self):
        """Test empty documents handling."""
        results = smart_search("test", [])
        assert len(results) == 0
    
    def test_no_matches(self):
        """Test query with no matches."""
        documents = ["The cat sits on the mat"]
        results = smart_search("elephant", documents)
        assert len(results) == 0


class TestSmartSearchEngine:
    """Test suite for SmartSearchEngine class."""
    
    def setup_method(self):
        """Setup test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def teardown_method(self):
        """Cleanup test database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = SmartSearchEngine(self.db_path)
        assert engine.db_path == self.db_path
        assert engine.k1 == 1.5
        assert engine.b == 0.75
        engine.close()
    
    def test_custom_parameters(self):
        """Test engine with custom parameters."""
        engine = SmartSearchEngine(
            self.db_path, 
            k1=2.0, 
            b=0.5, 
            context_weight=0.4
        )
        assert engine.k1 == 2.0
        assert engine.b == 0.5
        assert engine.context_weight == 0.4
        engine.close()
    
    def test_index_and_search(self):
        """Test document indexing and searching."""
        documents = [
            "Machine learning is artificial intelligence",
            "Python is a programming language",
            "Search engines use algorithms"
        ]
        
        engine = SmartSearchEngine(self.db_path)
        engine.index_documents(documents)
        
        results = engine.search("machine learning", top_k=5)
        assert len(results) > 0
        
        # Check that the most relevant document is first
        doc_id, score, content = results[0]
        assert "machine learning" in content.lower()
        
        engine.close()
    
    def test_explain_search(self):
        """Test search explanation functionality."""
        documents = [
            "Machine learning algorithms",
            "Deep learning networks"
        ]
        
        engine = SmartSearchEngine(self.db_path)
        engine.index_documents(documents)
        
        results = engine.search("machine learning")
        assert len(results) > 0
        
        doc_id = results[0][0]
        explanation = engine.explain_search("machine learning", doc_id)
        
        assert 'query' in explanation
        assert 'bm25_score' in explanation
        assert 'context_score' in explanation
        assert 'final_score' in explanation
        
        engine.close()
    
    def test_get_stats(self):
        """Test statistics functionality."""
        documents = ["Test document one", "Test document two"]
        
        engine = SmartSearchEngine(self.db_path)
        engine.index_documents(documents)
        
        stats = engine.get_stats()
        
        assert stats['total_documents'] == 2
        assert stats['total_unique_terms'] > 0
        assert 'features' in stats
        
        engine.close()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        documents = ["Test document"]
        
        with SmartSearchEngine(self.db_path) as engine:
            engine.index_documents(documents)
            results = engine.search("test")
            assert len(results) > 0
        
        # Engine should be closed automatically
    
    def test_invalid_document_type(self):
        """Test handling of invalid document types."""
        engine = SmartSearchEngine(self.db_path)
        
        with pytest.raises(TypeError):
            engine.index_documents([123, "valid document"])
        
        engine.close()
    
    def test_empty_documents_list(self):
        """Test handling of empty documents list."""
        engine = SmartSearchEngine(self.db_path)
        
        with pytest.raises(ValueError):
            engine.index_documents([])
        
        engine.close()


class TestMultilingual:
    """Test multilingual functionality."""
    
    def setup_method(self):
        """Setup test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def teardown_method(self):
        """Cleanup test database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_language_detection(self):
        """Test automatic language detection."""
        documents = [
            "The cat sits on the mat",  # English
            "Кот сидит на коврике"      # Russian
        ]
        
        engine = SmartSearchEngine(self.db_path)
        
        # Test language detection
        lang_en = engine._detect_language("Hello world")
        lang_ru = engine._detect_language("Привет мир")
        
        assert lang_en == 'en'
        assert lang_ru == 'ru'
        
        engine.close()
    
    def test_multilingual_search(self):
        """Test searching in multiple languages."""
        documents = [
            "The cat is sleeping",
            "Кот спит на диване",
            "Dogs are playing",
            "Собаки играют во дворе"
        ]
        
        engine = SmartSearchEngine(self.db_path)
        engine.index_documents(documents)
        
        # English search
        results_en = engine.search("cat sleeping")
        assert len(results_en) > 0
        
        # Russian search
        results_ru = engine.search("кот спит")
        assert len(results_ru) > 0
        
        engine.close()


if __name__ == "__main__":
    pytest.main([__file__])
