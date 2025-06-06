import pytest
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

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

def test_migration_file_exists():
    """Тест наличия файла миграции"""
    migration_file = Path('migrations/20240315_000000_create_tables.sql')
    assert migration_file.exists()
    assert migration_file.is_file()

def test_migration_file_content():
    """Тест содержимого файла миграции"""
    migration_file = Path('migrations/20240315_000000_create_tables.sql')
    content = migration_file.read_text()
    
    # Проверка наличия основных элементов
    assert 'BEGIN;' in content
    assert 'COMMIT;' in content
    assert 'CREATE TABLE' in content
    assert 'CREATE INDEX' in content
    
    # Проверка наличия всех таблиц
    assert 'CREATE TABLE IF NOT EXISTS satellite_data' in content
    assert 'CREATE TABLE IF NOT EXISTS seismic_data' in content
    assert 'CREATE TABLE IF NOT EXISTS eruptions' in content
    assert 'CREATE TABLE IF NOT EXISTS predictions' in content

def test_migration_application(config, db_params):
    """Тест применения миграции"""
    # Команда для применения миграции
    cmd = [
        'psql',
        f"--dbname={db_params['dbname']}",
        f"--username={db_params['user']}",
        f"--host={db_params['host']}",
        f"--port={db_params['port']}",
        "--file=migrations/20240315_000000_create_tables.sql"
    ]
    
    # Установка переменной окружения для пароля
    env = os.environ.copy()
    env['PGPASSWORD'] = db_params['password']
    
    try:
        # Выполнение команды
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert result.returncode == 0, f"Ошибка применения миграции: {result.stderr}"
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Ошибка при применении миграции: {str(e)}")

def test_table_creation(config, db_params):
    """Тест создания таблиц"""
    # Команда для проверки существования таблиц
    cmd = [
        'psql',
        f"--dbname={db_params['dbname']}",
        f"--username={db_params['user']}",
        f"--host={db_params['host']}",
        f"--port={db_params['port']}",
        "--command=SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
    ]
    
    # Установка переменной окружения для пароля
    env = os.environ.copy()
    env['PGPASSWORD'] = db_params['password']
    
    try:
        # Выполнение команды
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert result.returncode == 0, f"Ошибка проверки таблиц: {result.stderr}"
        
        # Проверка наличия всех таблиц
        output = result.stdout.lower()
        assert 'satellite_data' in output
        assert 'seismic_data' in output
        assert 'eruptions' in output
        assert 'predictions' in output
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Ошибка при проверке таблиц: {str(e)}")

def test_index_creation(config, db_params):
    """Тест создания индексов"""
    # Команда для проверки существования индексов
    cmd = [
        'psql',
        f"--dbname={db_params['dbname']}",
        f"--username={db_params['user']}",
        f"--host={db_params['host']}",
        f"--port={db_params['port']}",
        "--command=SELECT indexname FROM pg_indexes WHERE schemaname='public';"
    ]
    
    # Установка переменной окружения для пароля
    env = os.environ.copy()
    env['PGPASSWORD'] = db_params['password']
    
    try:
        # Выполнение команды
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert result.returncode == 0, f"Ошибка проверки индексов: {result.stderr}"
        
        # Проверка наличия всех индексов
        output = result.stdout.lower()
        assert 'idx_satellite_timestamp' in output
        assert 'idx_seismic_timestamp' in output
        assert 'idx_eruptions_start_time' in output
        assert 'idx_predictions_timestamp' in output
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Ошибка при проверке индексов: {str(e)}")

def test_column_types(config, db_params):
    """Тест типов столбцов"""
    # Команда для проверки типов столбцов
    cmd = [
        'psql',
        f"--dbname={db_params['dbname']}",
        f"--username={db_params['user']}",
        f"--host={db_params['host']}",
        f"--port={db_params['port']}",
        """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema='public' 
        AND table_name='predictions';
        """
    ]
    
    # Установка переменной окружения для пароля
    env = os.environ.copy()
    env['PGPASSWORD'] = db_params['password']
    
    try:
        # Выполнение команды
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert result.returncode == 0, f"Ошибка проверки типов столбцов: {result.stderr}"
        
        # Проверка типов столбцов
        output = result.stdout.lower()
        assert 'probability' in output and 'double precision' in output
        assert 'threshold' in output and 'double precision' in output
        assert 'is_eruption_predicted' in output and 'boolean' in output
        assert 'feature_importance' in output and 'jsonb' in output
        assert 'temporal_importance' in output and 'jsonb' in output
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Ошибка при проверке типов столбцов: {str(e)}") 