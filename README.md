# Система предсказания извержений вулканов

## Описание проекта
Система для предсказания извержений вулканов на основе спутниковых и сейсмических данных с использованием модели Temporal Fusion Transformer.

## Требования
- Docker и Docker Compose
- Python 3.8+
- PostgreSQL 13+
- GDAL библиотеки

## Установка и запуск

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Запуск с использованием Docker
```bash
# Сборка и запуск контейнеров
docker-compose up --build

# Запуск в фоновом режиме
docker-compose up -d
```

### 3. Проверка работоспособности
- Веб-интерфейс PgAdmin доступен по адресу: http://localhost:5050
  - Email: admin@admin.com
  - Пароль: admin
- База данных PostgreSQL доступна на порту 5432

## Структура проекта
```
.
├── config/                 # Конфигурационные файлы
├── data/                   # Директория для данных
│   ├── satellite_images/   # Спутниковые изображения
│   └── seismic/           # Сейсмические данные
├── src/                    # Исходный код
│   ├── data_processing/   # Обработка данных
│   ├── models/            # Модели
│   ├── training/          # Обучение моделей
│   └── database/          # Работа с базой данных
├── tests/                 # Тесты
├── migrations/            # Миграции базы данных
├── scripts/              # Скрипты
├── checkpoints/          # Чекпоинты моделей
└── logs/                 # Логи
```

## Использование

### Обучение модели
```bash
python train.py --config config/model_config.json
```

### Предсказание
```bash
python predict.py --config config/model_config.json --input data/input_data.json
```

### Сбор данных
```bash
python run_collector.py
```

## Тестирование
```bash
# Запуск всех тестов
pytest

# Запуск тестов с отчетом о покрытии
pytest --cov=src tests/
```

## Мониторинг

### Метрики системы
- Точность предсказаний (Accuracy, Precision, Recall, F1-score)
- Время отклика системы (Response Time)
- Использование ресурсов (CPU, Memory, Disk)
- Качество данных (Data Quality Metrics)

### Алерты
- Критические изменения в данных
- Ошибки в работе системы
- Проблемы с подключением к БД
- Аномалии в предсказаниях

### Логи
- Логи приложения: `logs/app.log`
- Логи базы данных: `logs/database.log`
- Метрики обучения: `logs/training.log`
- Логи мониторинга: `logs/monitoring.log`

## CI/CD

### GitHub Actions
- Автоматическое тестирование при пуше
- Проверка качества кода
- Сборка и публикация Docker-образов
- Автоматическое развертывание

### Качество кода
- Проверка стиля кода (flake8)
- Проверка типов (mypy)
- Тестовое покрытие (pytest-cov)
- Безопасность (bandit)

## Резервное копирование
```bash
# Создание резервной копии
python scripts/backup_db.py

# Восстановление из резервной копии
python scripts/restore_db.py --backup <backup-file>
```

## Лицензия
MIT

## Авторы
- [Ваше имя]

## Благодарности
- Ключевская вулканическая станция
- Институт вулканологии и сейсмологии ДВО РАН 