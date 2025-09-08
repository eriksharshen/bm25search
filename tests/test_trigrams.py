#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bm25search import SmartSearchEngine


def test_trigram_queries():
    """Расширенный тест триграмм: большой корпус, RU/EN запросы, explain по топу."""

    engine = SmartSearchEngine(
        db_path=":memory:",
        # базовые BM25
        k1=1.5,
        b=0.75,
        # веса и бонусы
        original_weight=1.0,
        normalized_weight=0.6,
        exact_surface_bonus=0.6,
        # фразы и близость
        use_bigrams=True,
        use_trigrams=True,
        phrase_weight=1.3,
        trigram_weight=1.8,
        proximity_bonus=0.4,
        proximity_window=4,
        # индекс по умолчанию без df-порога, можно поднять до 2
        min_ngram_df=1,
    )

    # Более обширный корпус (RU + EN)
    documents = [
        # RU — управление проектами / аналитика
        "Этот детальный план проекта согласован с командой и заказчиком",
        "Мы подготовили дорожную карту проекта на следующий квартал",
        "Команда работает над анализом данных для нового отчета",
        "План проекта включает оценку рисков и бюджетирование",
        "Согласован план проекта и утвержден график работ",
        "Разработка модуля авторизации завершена успешно",
        "Результаты аналитики показали рост конверсии",
        # EN — IR / DL / ML
        "We use BM25 ranking for information retrieval tasks",
        "A detailed project plan was aligned with all stakeholders",
        "Neural networks for text processing require large datasets",
        "The project management plan includes risk assessment",
        "Information retrieval system performance was improved",
        "Deep learning models benefit from regularization and data augmentation",
        # Микс RU/EN
        "Наши проекты включают NLP и information retrieval",
        "Проектный офис ведет проектный план и бэклог",
    ]

    engine.index_documents(documents)

    # Разнообразные запросы (ориентированные на триграммы)
    queries = [
        # RU триграммы/фразы
        "детальный план проекта согласован",  # включает триграмму "детальный план проекта"
        "план проекта включает оценку",       # включает триграмму "план проекта включает"
        "согласован план проекта",            # фраза и биграммы
        # EN триграммы/фразы
        "project management plan includes",   # триграмма "project management plan"
        "information retrieval system performance",  # триграмма
        # Смешанные запросы
        "детальный project plan",             # смешанный
    ]

    for q in queries:
        print(f"Поиск: {q}")
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

    # Объяснение по двум характерным запросам
    explain_samples = [
        "детальный план проекта согласован",
        "information retrieval system performance",
    ]

    for q in explain_samples:
        results = engine.search(q, top_k=1)
        if not results:
            continue
        best_id, score, content = results[0]
        info = engine.explain_search(q, best_id)
        print("Объяснение:")
        print(f"  Запрос: {q}")
        print(f"  Документ: {content}")
        print(f"  Итоговый скор: {info['final_score']:.2f}")
        print(f"  Точные совпадения: {info.get('exact_surface_matches', [])} (+{info.get('exact_surface_bonus', 0.0)})")
        # Показать n-граммы
        bigrams = [(t, d) for t, d in info['term_details'].items() if d.get('type') == 'bigram']
        trigrams = [(t, d) for t, d in info['term_details'].items() if d.get('type') == 'trigram']
        if bigrams:
            print("  Фразы (биграммы):")
            for t, d in bigrams:
                print(f"    {t}: tf={d['tf']}, idf={d['idf']:.2f}, w={d['weight']}, score={d['score']:.2f}")
        if trigrams:
            print("  Фразы (триграммы):")
            for t, d in trigrams:
                print(f"    {t}: tf={d['tf']}, idf={d['idf']:.2f}, w={d['weight']}, score={d['score']:.2f}")
        # Proximity
        if info.get('proximity_pairs'):
            print("  Proximity пары:")
            for p in info['proximity_pairs']:
                print(f"    {p['pair']} -> dist={p['min_distance_chars']}, bonus={p['bonus']:.2f}")
        print()

    engine.close()


if __name__ == "__main__":
    test_trigram_queries()
