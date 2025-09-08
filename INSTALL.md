# Установка и использование BM25Search

## 📦 Установка

### Из GitHub (рекомендуется)
```bash
# Базовая установка
pip install git+https://github.com/yourusername/bm25search.git

# С полными зависимостями (NLTK, pymorphy2)
pip install git+https://github.com/yourusername/bm25search.git[full]
```

### Для разработки
```bash
git clone https://github.com/yourusername/bm25search.git
cd bm25search
pip install -e .[dev]
```

### Из PyPI (если опубликован)
```bash
pip install bm25search
```

## 🚀 Быстрый старт

### Простой поиск
```python
from bm25search import smart_search

documents = [
    "Python - мощный язык программирования",
    "JavaScript используется для веб-разработки", 
    "SQL нужен для работы с базами данных"
]

results = smart_search(documents, "программирование")
for doc_id, score, content in results:
    print(f"{score:.2f} - {content}")
```

### Продвинутое использование
```python
from bm25search import SmartSearchEngine

# Создание поисковика
engine = SmartSearchEngine(
    db_path="my_search.db",  # Файл базы данных
    k1=1.5,                  # Параметр BM25
    b=0.75                   # Параметр BM25
)

# Индексация документов
documents = [
    "Машинное обучение и искусственный интеллект",
    "Нейронные сети для глубокого обучения",
    "Алгоритмы классификации и регрессии"
]

engine.index_documents(documents)

# Поиск
results = engine.search("обучение", top_k=5)
for doc_id, score, content in results:
    print(f"{score:.2f} - {content}")

# Объяснение результатов
explanation = engine.explain_search("обучение", doc_id=0)
print(f"BM25 скор: {explanation['bm25_score']:.3f}")

# Статистика
stats = engine.get_stats()
print(f"Документов: {stats['total_documents']}")
print(f"Уникальных терминов: {stats['total_unique_terms']}")

# Закрытие
engine.close()
```

## 🔧 Зависимости

### Обязательные
- Python >= 3.7

### Опциональные (устанавливаются с `[full]`)
- `nltk` - для стемминга английского языка
- `pymorphy2` - для лемматизации русского языка
- `langdetect` - для автоопределения языка

## 📁 Структура проекта

```
bm25search/
├── bm25search/
│   ├── __init__.py      # Основные импорты
│   ├── core.py          # Главный поисковик
│   └── utils.py         # Утилиты
├── examples/
│   └── basic_usage.py   # Примеры использования
├── tests/
│   └── test_basic.py    # Тесты
├── setup.py             # Установка
├── requirements.txt     # Зависимости
└── README.md           # Документация
```

## 🌍 Многоязычность

Поисковик автоматически определяет язык и применяет соответствующую обработку:

```python
# Русский текст - лемматизация через pymorphy2
engine.search("программирование")  # найдет "программа", "программист"

# Английский текст - стемминг через NLTK
engine.search("programming")       # найдет "program", "programmer"
```

## ⚡ Производительность

- **SQLite** для персистентного хранения индекса
- **Быстрая индексация** - документы обрабатываются один раз
- **Эффективный поиск** - BM25 с оптимизированными запросами к БД

## 🔍 Примеры запуска

```bash
# Запуск примеров
cd bm25search
python examples/basic_usage.py

# Запуск тестов
python -m pytest tests/

# Проверка установки
python -c "from bm25search import SmartSearchEngine; print('OK')"
```

## 📝 Лицензия

MIT License - можно использовать в коммерческих проектах.
