#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bm25search import SmartSearchEngine

def test_sentence_queries():
    """Тест: индексируем документы и ищем предложениями (RU и EN)."""

    # Настроим движок с легким бустом точной формы
    engine = SmartSearchEngine(db_path=":memory:", original_weight=1.0, normalized_weight=0.6, exact_surface_bonus=0.6)

    # Корпус документов (смешанный RU/EN)
    documents = [
        "Нейронные сети сегодня применяются в компьютерном зрении и обработке текста",
        "Проект по машинному обучению завершён успешно",
        "Наши проекты связаны с анализом данных и NLP",
        "Этот детальный план проекта согласован с командой",
        "Deep learning models require large datasets and careful regularization",
        "We use BM25 ranking for information retrieval",
        "Project management is crucial for timely delivery",
    ]

    print(" Документы:")
    for i, doc in enumerate(documents):
        print(f"  {i}: {doc}")
    print()

    engine.index_documents(documents)

    # Поисковые запросы-предложения (RU и EN)
    queries = [
        "проект по анализу данных",                 # RU фраза
        "детальный план проекта",                   # RU фраза с точной формой
        "мы работаем над проектами в nlp",         # RU с заимствованием
        "neural networks for text processing",      # EN
        "bm25 for document ranking",               # EN фраза
    ]

    for q in queries:
        print(f" Поиск: {q}")
        lang = engine._detect_language(q)
        processed, _ = engine._preprocess_text(q, lang)
        print(f"  Язык: {lang}")
        print(f"  Обработанные термины: {processed}")

        results = engine.search(q, top_k=5)
        if results:
            for doc_id, score, content in results:
                print(f"    {score:.2f} - {content}")
        else:
            print("    Ничего не найдено")
        print()

    # Пояснение для одного запроса и документа
    query = "детальный план проекта"
    results = engine.search(query, top_k=1)
    if results:
        best_id, score, content = results[0]
        info = engine.explain_search(query, best_id)
        print(" Объяснение лучшего результата:")
        print(f"  Документ: {content}")
        print(f"  Итоговый скор: {info['final_score']:.2f}")
        print(f"  Язык запроса: {info['query_language']} | Язык документа: {info['document_language']}")
        print(f"  Точные совпадения: {info.get('exact_surface_matches', [])} (+{info.get('exact_surface_bonus', 0.0)})")
        print("  Вклад термов:")
        for t, d in info['term_details'].items():
            print(f"    {t}: tf={d['tf']}, idf={d['idf']:.2f}, weight={d['weight']}, original={d['is_original']}, score={d['score']:.2f}")
        print()

    engine.close()

if __name__ == "__main__":
    test_sentence_queries()
