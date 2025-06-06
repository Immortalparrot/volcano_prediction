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

@pytest.fixture
def migration_dir():
    """Фикстура с директорией для миграций"""
    migration_dir = Path('migrations')
    migration_dir.mkdir(exist_ok=True)
    return migration_dir

def test_create_migration_file(migration_dir):
    """Тест создания файла миграции"""
    from scripts.migrate_db import create_migration_file
    
    # Создание файла миграции
    description = "test_migration"
    migration_file = create_migration_file(migration_dir, description)
    
    try:
        # Проверка наличия файла
        assert migration_file.exists()
        assert migration_file.is_file()
        
        # Проверка содержимого файла
        content = migration_file.read_text()
        assert '-- Миграция базы данных' in content
        assert f'-- Описание: {description}' in content
        assert 'BEGIN;' in content
        assert 'COMMIT;' in content
        
    finally:
        # Удаление тестового файла
        if migration_file.exists():
            migration_file.unlink()

def test_apply_migration(config, db_params, migration_dir):
    """Тест применения миграции"""
    from scripts.migrate_db import apply_migration
    
    # Создание тестового файла миграции
    migration_file = migration_dir / 'test_migration.sql'
    migration_file.write_text("""
        BEGIN;
        
        CREATE TABLE IF NOT EXISTS test_migration (
            id SERIAL PRIMARY KEY,
            test_field TEXT
        );
        
        COMMIT;
    """)
    
    try:
        # Применение миграции
        logger = logging.getLogger(__name__)
        assert apply_migration(config, migration_file, logger)
        
        # Проверка создания таблицы
        cmd = [
            'psql',
            f"--dbname={db_params['dbname']}",
            f"--username={db_params['user']}",
            f"--host={db_params['host']}",
            f"--port={db_params['port']}",
            "--command=SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'test_migration');"
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = db_params['password']
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert result.returncode == 0
        assert 't' in result.stdout.lower()
        
    finally:
        # Удаление тестовой таблицы и файла миграции
        if migration_file.exists():
            migration_file.unlink()
        
        # Удаление тестовой таблицы
        drop_cmd = [
            'psql',
            f"--dbname={db_params['dbname']}",
            f"--username={db_params['user']}",
            f"--host={db_params['host']}",
            f"--port={db_params['port']}",
            "--command=DROP TABLE IF EXISTS test_migration;"
        ]
        subprocess.run(drop_cmd, env=env, capture_output=True)

def test_get_pending_migrations(migration_dir):
    """Тест получения списка непримененных миграций"""
    from scripts.migrate_db import get_pending_migrations
    
    # Создание тестовых файлов миграций
    migration_files = [
        migration_dir / '20240315_000000_test1.sql',
        migration_dir / '20240315_000001_test2.sql',
        migration_dir / '20240315_000002_test3.sql'
    ]
    
    try:
        # Создание файлов
        for file in migration_files:
            file.write_text('-- Test migration\n')
        
        # Получение списка миграций
        pending_migrations = get_pending_migrations(migration_dir)
        
        # Проверка результатов
        assert len(pending_migrations) == 3
        assert all(file.name in [m.name for m in pending_migrations] for file in migration_files)
        
    finally:
        # Удаление тестовых файлов
        for file in migration_files:
            if file.exists():
                file.unlink()

def test_migration_rollback(config, db_params, migration_dir):
    """Тест отката миграции"""
    from scripts.migrate_db import apply_migration
    
    # Создание тестового файла миграции с ошибкой
    migration_file = migration_dir / 'test_rollback.sql'
    migration_file.write_text("""
        BEGIN;
        
        CREATE TABLE IF NOT EXISTS test_rollback (
            id SERIAL PRIMARY KEY,
            test_field TEXT
        );
        
        -- Намеренная ошибка
        SELECT * FROM non_existent_table;
        
        COMMIT;
    """)
    
    try:
        # Применение миграции
        logger = logging.getLogger(__name__)
        assert not apply_migration(config, migration_file, logger)
        
        # Проверка, что таблица не создана
        cmd = [
            'psql',
            f"--dbname={db_params['dbname']}",
            f"--username={db_params['user']}",
            f"--host={db_params['host']}",
            f"--port={db_params['port']}",
            "--command=SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'test_rollback');"
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = db_params['password']
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert result.returncode == 0
        assert 'f' in result.stdout.lower()
        
    finally:
        # Удаление тестового файла миграции
        if migration_file.exists():
            migration_file.unlink()

def test_migration_ordering(migration_dir):
    """Тест порядка применения миграций"""
    from scripts.migrate_db import get_pending_migrations
    
    # Создание тестовых файлов миграций с разными временными метками
    migration_files = [
        migration_dir / '20240315_000000_old.sql',
        migration_dir / '20240315_000001_middle.sql',
        migration_dir / '20240315_000002_new.sql'
    ]
    
    try:
        # Создание файлов
        for file in migration_files:
            file.write_text('-- Test migration\n')
        
        # Получение списка миграций
        pending_migrations = get_pending_migrations(migration_dir)
        
        # Проверка порядка
        assert len(pending_migrations) == 3
        assert pending_migrations[0].name == '20240315_000000_old.sql'
        assert pending_migrations[1].name == '20240315_000001_middle.sql'
        assert pending_migrations[2].name == '20240315_000002_new.sql'
        
    finally:
        # Удаление тестовых файлов
        for file in migration_files:
            if file.exists():
                file.unlink() 