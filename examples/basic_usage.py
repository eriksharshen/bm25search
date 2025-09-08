"""
Примеры использования BM25Search - чистый BM25 с стеммингом и лемматизацией.
"""

from bm25search import SmartSearchEngine, smart_search, batch_search

def quick_search_example():
    """Пример быстрого поиска без создания движка."""
    print("🔍 Быстрый поиск:")
    
    documents = [
        "Кот сидит на окне и смотрит на улицу",
        "Собака играет в парке с детьми", 
        "Красивая кошка спит на диване",
        "Умная собака выполняет команды хозяина"
    ]
    
    query = "кот"
    results = smart_search(documents, query, top_k=2)
    
    print(f"Запрос: '{query}'")
    for doc_id, score, content in results:
        print(f"  {score:.2f} - {content}")
    print()

def advanced_search_example():
    """Пример продвинутого поиска с настройками."""
    print("⚙️ Продвинутый поиск:")
    
    # Создаем движок с настройками
    engine = SmartSearchEngine(
        db_path="advanced_demo.db",
        k1=1.2,  # Менее агрессивное насыщение по частоте
        b=0.8    # Больший вес длины документа
    )
    
    documents = [
        "Машинное обучение - это подраздел искусственного интеллекта",
        "Нейронные сети используются для глубокого обучения",
        "Алгоритмы классификации решают задачи категоризации",
        "Регрессионный анализ предсказывает числовые значения"
    ]
    
    engine.index_documents(documents)
    
    query = "обучение"
    results = engine.search(query, top_k=3)
    
    print(f"Запрос: '{query}'")
    for doc_id, score, content in results:
        print(f"  {score:.2f} - {content}")
    
    # Подробное объяснение
    print("\n📊 Объяснение поиска:")
    explanation = engine.explain_search(query, doc_id=0)
    print(f"BM25 скор: {explanation['bm25_score']:.3f}")
    print(f"Финальный скор: {explanation['final_score']:.3f}")
    
    engine.close()
    print()

def multilingual_search_example():
    """Пример многоязычного поиска."""
    print("🌍 Многоязычный поиск:")
    
    engine = SmartSearchEngine(db_path="multilingual_demo.db")
    
    documents = [
        "The cat sits on the windowsill",
        "Кот сидит на подоконнике", 
        "A dog plays in the garden",
        "Собака играет в саду",
        "Beautiful flowers bloom in spring",
        "Красивые цветы цветут весной"
    ]
    
    engine.index_documents(documents)
    
    # Поиск на русском
    results_ru = engine.search("кот", top_k=2)
    print("Поиск 'кот':")
    for doc_id, score, content in results_ru:
        print(f"  {score:.2f} - {content}")
    
    # Поиск на английском  
    results_en = engine.search("cat", top_k=2)
    print("\nПоиск 'cat':")
    for doc_id, score, content in results_en:
        print(f"  {score:.2f} - {content}")
    
    engine.close()
    print()

def stemming_lemmatization_example():
    """Пример работы стемминга и лемматизации."""
    print("🔧 Стемминг и лемматизация:")
    
    engine = SmartSearchEngine(db_path="stemming_demo.db")
    
    documents = [
        "Программист программирует программы",
        "Разработчик разрабатывает приложения",
        "Кодер кодит код на Python",
        "Инженер проектирует системы"
    ]
    
    engine.index_documents(documents)
    
    # Поиск по разным формам слова
    queries = ["программа", "программирование", "разработка", "код"]
    
    for query in queries:
        results = engine.search(query, top_k=2)
        print(f"Запрос '{query}':")
        for doc_id, score, content in results:
            print(f"  {score:.2f} - {content}")
        print()
    
    engine.close()

def batch_search_example():
    """Пример пакетного поиска."""
    print("📦 Пакетный поиск:")
    
    documents = [
        "Python - мощный язык программирования",
        "JavaScript используется для веб-разработки",
        "SQL нужен для работы с базами данных",
        "HTML создает структуру веб-страниц"
    ]
    
    queries = ["Python", "веб", "данные"]
    
    all_results = batch_search(documents, queries, top_k=2)
    
    for query, results in all_results.items():
        print(f"Запрос '{query}':")
        for doc_id, score, content in results:
            print(f"  {score:.2f} - {content}")
        print()

def main():
    """Запуск всех примеров."""
    print("🚀 Примеры использования BM25Search (чистый BM25)\n")
    
    quick_search_example()
    advanced_search_example() 
    multilingual_search_example()
    stemming_lemmatization_example()
    batch_search_example()
    
    print("✅ Все примеры выполнены!")

if __name__ == "__main__":
    main()
