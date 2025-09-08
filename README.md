# BM25Search

[![PyPI version](https://badge.fury.io/py/bm25search.svg)](https://badge.fury.io/py/bm25search)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Чистый BM25 поисковый движок с морфологией (RU/EN), фразами и учётом близости**

## Описание

BM25Search — это простой и эффективный поисковый движок на основе BM25 с поддержкой:
- лемматизации (русский) и стемминга (английский),
- фраз (биграмм) с отдельным весом,
- бонуса за точные поверхностные совпадения,
- бонуса близости терминов (proximity).

## Особенности

- 🔍 **Чистый BM25**: Классический алгоритм ранжирования (IDF может быть отрицательным для частых терминов)
- 🗄️ **Персистентное хранение**: SQLite база данных для масштабируемости
- 🌍 **Многоязычность**: RU/EN с автоопределением языка (langid → langdetect → эвристика)
- 📝 **Морфология**: RU — лемматизация (pymorphy2), EN — стемминг (Snowball)
- 🧩 **Фразы (биграммы)**: Индексация биграмм и фразовый вклад с отдельным весом
- 📏 **Близость**: Proximity-бонус за близко расположенные термы запроса в документе
- 🛠️ **Веса**: Разные веса для точных форм, нормализованных форм и фраз
- 🚫 **Стоп-слова**: Автоматическая фильтрация с fallback-наборами
- 🔧 **Простой API**: Интуитивно понятный интерфейс и explain()
- 📦 **Единый модуль**: Вся функциональность в одном файле

## Установка

### Из GitHub

```bash
pip install git+https://github.com/yourusername/bm25search.git
```

### Для разработки

```bash
git clone https://github.com/yourusername/bm25search.git
cd bm25search
pip install -e .
```

### Локальная установка

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/bm25search.git
cd bm25search

# Установите в режиме разработки
pip install -e .

# Или установите зависимости отдельно
pip install -r requirements.txt
```

## Структура проекта

```
bm25search/
├── src/
│   └── bm25search.py          # Единый модуль со всей функциональностью
├── examples/
│   └── basic_usage.py         # Примеры использования
├── tests/
│   └── test_basic.py          # Базовые тесты
├── requirements.txt           # Зависимости
├── setup.py                   # Настройки установки
├── pyproject.toml            # Современная конфигурация пакета
└── README.md                 # Документация
```

## Зависимости

- `nltk>=3.6` — стемминг и стоп-слова
- `pymorphy2>=0.9.1` и `pymorphy2-dicts-ru` — русская лемматизация
- `langid>=1.1.6` — определение языка (основной)
- `langdetect>=1.0.9` — fallback определения языка
- Рекомендуется `setuptools` (для корректной инициализации pymorphy2)

## Быстрый старт

### Простой поиск

```python
from bm25search import smart_search

documents = [
    "Кот сидит на окне",
    "Собака бегает в парке", 
    "Python - мощный язык программирования"
]

results = smart_search(documents, "кот", top_k=5)
print(results[0])  # (0, 2.45, "Кот сидит на окне")
```

### Продвинутое использование

```python
from bm25search import SmartSearchEngine

# Создание движка
engine = SmartSearchEngine(
    db_path="my_search.db",
    k1=1.5,
    b=0.75,
    # Веса терминов и бонусы
    original_weight=1.0,       # вес точной формы запроса (поверхностной)
    normalized_weight=0.6,     # вес нормализованных форм (стем/лемма)
    exact_surface_bonus=0.5,   # бонус за точное совпадение слова в документе
    # Фразы и близость
    use_bigrams=True,          # индексация и учёт биграмм
    phrase_weight=1.3,         # вес фразового компонента (BM25 по биграмме)
    proximity_bonus=0.3,       # бонус за близость разных токенов запроса
    proximity_window=3         # окно близости (в словах, эвристически переводится в символы)
)

# Индексация документов
documents = [
    "Машинное обучение - это подраздел ИИ",
    "Python используется для анализа данных",
    "Нейронные сети решают сложные задачи"
]

engine.index_documents(documents)

# Поиск
results = engine.search("машинное обучение", top_k=3)
for doc_id, score, content in results:
    print(f"{score:.2f} - {content}")

# Объяснение скоринга
explanation = engine.explain_search("машинное обучение", doc_id=0)
print(f"BM25 скор: {explanation['bm25_score']:.3f}")

# Статистика
stats = engine.get_stats()
print(f"Документов: {stats['total_documents']}")
print(f"Уникальных терминов: {stats['total_unique_terms']}")

engine.close()
```

### Поиск предложениями и объяснение

```python
from bm25search import SmartSearchEngine

docs = [
    "Этот детальный план проекта согласован с командой",
    "Наши проекты связаны с анализом данных и NLP",
    "We use BM25 ranking for information retrieval",
]

with SmartSearchEngine() as engine:
    engine.index_documents(docs)

    q = "детальный план проекта"
    results = engine.search(q, top_k=3)
    for doc_id, score, content in results:
        print(f"{score:.2f} - {content}")

    # Объяснение лучшего результата
    best_id = results[0][0]
    info = engine.explain_search(q, best_id)
    print("Точные совпадения:", info.get("exact_surface_matches"))
    print("Фразы:", [t for t, d in info["term_details"].items() if d.get("type") == "bigram"])
    print("Proximity:", info.get("proximity_pairs"))
```

### Контекстный менеджер

```python
with SmartSearchEngine() as engine:
    engine.index_documents(documents)
    results = engine.search("запрос")
    # Автоматическое закрытие соединения
```

## Тестирование установки

После установки проверьте работоспособность:

```python
# Проверка импорта
import bm25search
print("✅ Импорт успешен!")

# Проверка создания движка
from bm25search import SmartSearchEngine
engine = SmartSearchEngine()
print("✅ Создание движка работает!")

# Быстрый тест
from bm25search import smart_search
results = smart_search(["тест документ"], "тест")
print(f"✅ Поиск работает: {len(results)} результатов")
```

Или запустите пример:

```bash
python examples/basic_usage.py
```

## API Справочник

### SmartSearchEngine

Основной класс поискового движка.

#### Конструктор

```python
SmartSearchEngine(
    db_path: str = "search_index.db",
    k1: float = 1.5,
    b: float = 0.75,
    original_weight: float = 1.0,
    normalized_weight: float = 0.6,
    exact_surface_bonus: float = 0.5,
    use_bigrams: bool = True,
    phrase_weight: float = 1.3,
    proximity_bonus: float = 0.3,
    proximity_window: int = 3,
)
```

#### Методы

- `index_documents(documents: List[str])` - индексация документов
- `search(query: str, top_k: int = 10)` - поиск по запросу
- `explain_search(query: str, doc_id: int)` - объяснение скоринга
- `get_stats()` - статистика движка
- `close()` - закрытие соединения с БД

### Утилитарные функции

- `smart_search(documents, query, top_k=10, **kwargs)` - быстрый поиск
- `batch_search(documents, queries, top_k=10, **kwargs)` - пакетный поиск

### Псевдонимы для совместимости

- `BM25Search` = `SmartSearchEngine`
- `quick_search` = `smart_search`

## Особенности реализации

### Единый модуль

Вся функциональность объединена в один файл `src/bm25search.py` для упрощения:
- Легче установка и распространение
- Меньше зависимостей между файлами
- Простая отладка и модификация
- Быстрый импорт

### Обработка текста

1. **Определение языка**: `langid` → `langdetect` → кириллическая эвристика
2. **Токенизация**: разбиение на слова (+ последовательность оригинальных токенов)
3. **Фильтрация**: удаление стоп-слов и коротких слов (fallback-наборы при отсутствии NLTK)
4. **Морфология**: RU — лемматизация (pymorphy2), EN — стемминг (Snowball)
5. **Индексация**:
   - униграммы (нормализованные термы) с позициями,
   - при `use_bigrams=True` — биграммы (из оригинальных токенов) с позициями.
6. **Ранжирование**:
   - BM25 по термам (с весами original/normalized),
   - BM25 по биграммам (с `phrase_weight`),
   - бонус за точное поверхностное совпадение (`exact_surface_bonus`),
   - proximity-бонус за близость оригинальных токенов запроса.

## Примеры использования

### Поиск в документации

```python
from bm25search import SmartSearchEngine

docs = [
    "Функция map применяет функцию к каждому элементу списка",
    "Метод filter отфильтровывает элементы по условию",
    "Генераторы списков позволяют создавать списки компактно"
]

with SmartSearchEngine() as engine:
    engine.index_documents(docs)
    results = engine.search("функция список")
    for doc_id, score, content in results:
        print(f"{score:.2f}: {content}")
```

### Многоязычный поиск

```python
docs = [
    "Кот сидит на подоконнике",
    "The cat sits on the windowsill"
]

with SmartSearchEngine() as engine:
    engine.index_documents(docs)
    
    print("Поиск 'кот':")
    for doc_id, score, content in engine.search("кот"):
        print(f"  {score:.2f} - {content}")
    
    print("Поиск 'cat':")
    for doc_id, score, content in engine.search("cat"):
        print(f"  {score:.2f} - {content}")
```

### Объяснение результатов поиска

```python
with SmartSearchEngine() as engine:
    engine.index_documents(docs)
    results = engine.search("кот")
    best_id = results[0][0]
    info = engine.explain_search("кот", best_id)
    print("Объяснение результатов поиска:")
    print("  - BM25 по термам:", info['bm25_term_scores'])
    print("  - BM25 по биграммам:", info['bm25_bigram_scores'])
    print("  - Бонус за точное совпадение:", info['exact_surface_bonus'])
    print("  - Proximity-бонус:", info['proximity_bonus'])
```

## Устранение неполадок

### Проблемы с установкой

```bash
# Обновите pip
python -m pip install --upgrade pip

# Установите зависимости отдельно (минимум)
pip install nltk pymorphy2 pymorphy2-dicts-ru langid langdetect setuptools

# Загрузите NLTK данные (при необходимости)
python -c "import nltk; nltk.download('stopwords')"
```

Если используете Python 3.11+ и сталкиваетесь с ошибками вида `inspect.getargspec` при инициализации pymorphy2,
в модуле уже предусмотрен совместимый шим. Убедитесь, что установлен `setuptools`.

### Проблемы с импортом

```python
# Проверьте установку
import sys
print(sys.path)

# Проверьте модуль
import bm25search
print(bm25search.__file__)
```

### Низкое качество поиска

```python
# Проверьте обработку текста
engine = SmartSearchEngine()
explanation = engine.explain_search("тест", 0)
print("Обработанные термины:", explanation['processed_query_terms'])
```

## Производительность

- **Индексация**: ~1000 документов/сек
- **Поиск**: ~100-1000 запросов/сек  
- **Память**: ~10MB на 10,000 документов
- **Диск**: SQLite файл ~1MB на 1,000 документов

## Лицензия

MIT License. См. [LICENSE](LICENSE) для подробностей.

## Поддержка

- 🐛 [GitHub Issues](https://github.com/yourusername/bm25search/issues)
- 📖 [Документация](https://github.com/yourusername/bm25search#readme)
