#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bm25search import SmartSearchEngine

def test_morphology():
    """Тестируем стемминг и лемматизацию для слова 'проект'"""
    
    engine = SmartSearchEngine(db_path=":memory:")
    
    # Проверим, что языковые процессоры загружены
    print(" Проверка языковых процессоров:")
    print(f"  Русский процессор: {engine.processors.get('ru', 'НЕ НАЙДЕН')}")
    if 'ru' in engine.processors:
        ru_proc = engine.processors['ru']
        print(f"    Стеммер: {ru_proc.get('stemmer', 'НЕТ')}")
        print(f"    Лемматизатор: {ru_proc.get('lemmatizer', 'НЕТ')}")
        
        # Тестируем стемминг и лемматизацию напрямую
        if ru_proc.get('stemmer'):
            stem_result = ru_proc['stemmer'].stem('проекты')
            print(f"    Стемминг 'проекты': {stem_result}")
        
        if ru_proc.get('lemmatizer'):
            lemma_result = ru_proc['lemmatizer'].parse('проекты')[0].normal_form
            print(f"    Лемматизация 'проекты': {lemma_result}")
    print()
    
    # Тестовые документы
    documents = [
        "У нас есть интересный проект по машинному обучению",
        "Мы работаем над несколькими проектами одновременно", 
        "В общем, был НС. На нем обсуждались КПД, в т.ч. эти ИИ-проекты",
        "Этот проектный план очень детальный"
    ]
    
    print(" Документы:")
    for i, doc in enumerate(documents):
        print(f"  {i}: {doc}")
    print()
    
    engine.index_documents(documents)
    
    # Тестируем разные формы слова
    test_queries = ["проект", "проекты", "проектов", "проектный"]
    
    for query in test_queries:
        print(f" Поиск '{query}':")
        
        # Показываем обработанные термины
        language = engine._detect_language(query)
        terms, positions = engine._preprocess_text(query, language)
        print(f"  Обработанные термины: {terms}")
        
        # Выполняем поиск
        results = engine.search(query, top_k=10)
        if results:
            for doc_id, score, content in results:
                print(f"    {score:.2f} - {content}")
        else:
            print("    Ничего не найдено")
        print()
    
    # Проверим индексированные термины для первого документа
    print(" Проверим, какие термины были проиндексированы:")
    cursor = engine.conn.execute('SELECT DISTINCT term FROM term_index ORDER BY term')
    all_terms = [row[0] for row in cursor.fetchall()]
    project_terms = [term for term in all_terms if 'проект' in term]
    print(f"  Термины с 'проект': {project_terms}")
    
    engine.close()

if __name__ == "__main__":
    test_morphology()