import pytest
import os
import json
import subprocess
from pathlib import Path
import logging

@pytest.fixture
def config():
    """Фикстура с конфигурацией базы данных"""
    config_path = Path('config/database_config.json')
    if not config_path.exists():
        raise FileNotFoundError("Файл конфигурации не найден")
    
    with open(config_path) as f:
        return json.load(f)

@pytest.fixture
def db_params(config):
    """Фикстура с параметрами подключения к базе данных"""
    return {
        'dbname': config['database']['name'],
        'user': config['database']['user'],
        'password': config['database']['password'],
        'host': config['database']['host'],
        'port': config['database']['port']
    }

def test_init_db_script():
    """Тест наличия скрипта инициализации базы данных"""
    init_script = Path('init_db.py')
    assert init_script.exists()
    assert init_script.is_file()

def test_create_tables(config, db_params):
    """Тест создания таблиц"""
    from init_db import create_tables
    
    # Создание временного соединения
    import psycopg2
    conn = psycopg2.connect(**db_params)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        # Создание таблиц
        create_tables(cursor)
        
        # Проверка создания таблиц
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'satellite_data' in tables
        assert 'seismic_data' in tables
        assert 'eruptions' in tables
        assert 'predictions' in tables
        
    finally:
        cursor.close()
        conn.close()

def test_create_indexes(config, db_params):
    """Тест создания индексов"""
    from init_db import create_tables
    
    # Создание временного соединения
    import psycopg2
    conn = psycopg2.connect(**db_params)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        # Создание таблиц и индексов
        create_tables(cursor)
        
        # Проверка создания индексов
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public'
        """)
        indexes = [row[0] for row in cursor.fetchall()]
        
        assert 'idx_satellite_timestamp' in indexes
        assert 'idx_seismic_timestamp' in indexes
        assert 'idx_eruptions_start_time' in indexes
        assert 'idx_predictions_timestamp' in indexes
        
    finally:
        cursor.close()
        conn.close()

def test_create_data_directories():
    """Тест создания директорий для данных"""
    from init_db import main
    
    # Запуск инициализации
    main()
    
    # Проверка создания директорий
    assert Path('data/satellite_images').exists()
    assert Path('data/seismic').exists()
    assert Path('data/eruptions').exists()

def test_database_connection(config, db_params):
    """Тест подключения к базе данных"""
    import psycopg2
    
    try:
        # Попытка подключения
        conn = psycopg2.connect(**db_params)
        assert conn is not None
        
        # Проверка возможности выполнения запросов
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        result = cursor.fetchone()
        assert result[0] == 1
        
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def test_table_columns(config, db_params):
    """Тест структуры таблиц"""
    from init_db import create_tables
    
    # Создание временного соединения
    import psycopg2
    conn = psycopg2.connect(**db_params)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        # Создание таблиц
        create_tables(cursor)
        
        # Проверка структуры таблицы satellite_data
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'satellite_data'
        """)
        columns = {row[0]: row[1] for row in cursor.fetchall()}
        
        assert 'id' in columns and columns['id'] == 'integer'
        assert 'timestamp' in columns and columns['timestamp'] == 'timestamp without time zone'
        assert 'ndvi' in columns and columns['ndvi'] == 'double precision'
        assert 'thermal_anomaly' in columns and columns['thermal_anomaly'] == 'boolean'
        
        # Проверка структуры таблицы predictions
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'predictions'
        """)
        columns = {row[0]: row[1] for row in cursor.fetchall()}
        
        assert 'id' in columns and columns['id'] == 'integer'
        assert 'timestamp' in columns and columns['timestamp'] == 'timestamp without time zone'
        assert 'probability' in columns and columns['probability'] == 'double precision'
        assert 'is_eruption_predicted' in columns and columns['is_eruption_predicted'] == 'boolean'
        assert 'feature_importance' in columns and columns['feature_importance'] == 'jsonb'
        
    finally:
        cursor.close()
        conn.close()

def test_logging_setup():
    """Тест настройки логирования"""
    from init_db import main
    
    # Запуск инициализации
    main()
    
    # Проверка наличия файла лога
    log_file = Path('logs/init_db.log')
    assert log_file.exists()
    assert log_file.is_file()
    
    # Проверка содержимого лога
    log_content = log_file.read_text()
    assert 'Подключение к базе данных' in log_content
    assert 'Создание таблиц' in log_content
    assert 'Инициализация базы данных успешно завершена' in log_content 