import pytest
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import gzip
import shutil

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
def backup_dir(config):
    """Фикстура с директорией для бэкапов"""
    backup_dir = Path(config['backup']['path'])
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir

def test_backup_creation(config, db_params, backup_dir):
    """Тест создания резервной копии"""
    # Команда для создания бэкапа
    cmd = [
        'pg_dump',
        f"--dbname={db_params['dbname']}",
        f"--username={db_params['user']}",
        f"--host={db_params['host']}",
        f"--port={db_params['port']}",
        f"--file={backup_dir}/test_backup.sql"
    ]
    
    # Установка переменной окружения для пароля
    env = os.environ.copy()
    env['PGPASSWORD'] = db_params['password']
    
    try:
        # Выполнение команды
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert result.returncode == 0, f"Ошибка создания бэкапа: {result.stderr}"
        
        # Проверка наличия файла бэкапа
        backup_file = backup_dir / 'test_backup.sql'
        assert backup_file.exists()
        assert backup_file.is_file()
        
        # Проверка содержимого бэкапа
        content = backup_file.read_text()
        assert 'CREATE TABLE' in content
        assert 'satellite_data' in content
        assert 'seismic_data' in content
        assert 'eruptions' in content
        assert 'predictions' in content
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Ошибка при создании бэкапа: {str(e)}")
    finally:
        # Удаление тестового бэкапа
        if backup_file.exists():
            backup_file.unlink()

def test_backup_compression(config, db_params, backup_dir):
    """Тест сжатия резервной копии"""
    # Создание тестового файла
    test_file = backup_dir / 'test_backup.sql'
    test_file.write_text('-- Test backup content\n')
    
    try:
        # Сжатие файла
        with open(test_file, 'rb') as f_in:
            with gzip.open(f'{test_file}.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Проверка сжатого файла
        compressed_file = backup_dir / 'test_backup.sql.gz'
        assert compressed_file.exists()
        assert compressed_file.is_file()
        
        # Проверка возможности распаковки
        with gzip.open(compressed_file, 'rb') as f:
            content = f.read().decode()
            assert 'Test backup content' in content
        
    finally:
        # Удаление тестовых файлов
        if test_file.exists():
            test_file.unlink()
        if compressed_file.exists():
            compressed_file.unlink()

def test_backup_cleanup(config, backup_dir):
    """Тест очистки старых резервных копий"""
    # Создание тестовых файлов бэкапов
    old_date = datetime.now() - timedelta(days=config['backup']['retention_days'] + 1)
    new_date = datetime.now() - timedelta(days=1)
    
    old_backup = backup_dir / f"backup_{old_date.strftime('%Y%m%d_%H%M%S')}.sql.gz"
    new_backup = backup_dir / f"backup_{new_date.strftime('%Y%m%d_%H%M%S')}.sql.gz"
    
    try:
        # Создание тестовых файлов
        old_backup.write_bytes(b'old backup')
        new_backup.write_bytes(b'new backup')
        
        # Запуск скрипта очистки
        from scripts.backup_db import cleanup_old_backups
        cleanup_old_backups(config, logging.getLogger(__name__))
        
        # Проверка результатов
        assert not old_backup.exists()
        assert new_backup.exists()
        
    finally:
        # Удаление тестовых файлов
        if old_backup.exists():
            old_backup.unlink()
        if new_backup.exists():
            new_backup.unlink()

def test_backup_restore(config, db_params, backup_dir):
    """Тест восстановления из резервной копии"""
    # Создание тестового бэкапа
    backup_file = backup_dir / 'test_restore.sql'
    backup_file.write_text("""
        CREATE TABLE IF NOT EXISTS test_restore (
            id SERIAL PRIMARY KEY,
            test_field TEXT
        );
    """)
    
    try:
        # Команда для восстановления
        cmd = [
            'psql',
            f"--dbname={db_params['dbname']}",
            f"--username={db_params['user']}",
            f"--host={db_params['host']}",
            f"--port={db_params['port']}",
            f"--file={backup_file}"
        ]
        
        # Установка переменной окружения для пароля
        env = os.environ.copy()
        env['PGPASSWORD'] = db_params['password']
        
        # Выполнение команды
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert result.returncode == 0, f"Ошибка восстановления: {result.stderr}"
        
        # Проверка создания таблицы
        check_cmd = [
            'psql',
            f"--dbname={db_params['dbname']}",
            f"--username={db_params['user']}",
            f"--host={db_params['host']}",
            f"--port={db_params['port']}",
            "--command=SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'test_restore');"
        ]
        
        check_result = subprocess.run(check_cmd, env=env, capture_output=True, text=True)
        assert check_result.returncode == 0
        assert 't' in check_result.stdout.lower()
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Ошибка при восстановлении: {str(e)}")
    finally:
        # Удаление тестовой таблицы и файла бэкапа
        if backup_file.exists():
            backup_file.unlink()
        
        # Удаление тестовой таблицы
        drop_cmd = [
            'psql',
            f"--dbname={db_params['dbname']}",
            f"--username={db_params['user']}",
            f"--host={db_params['host']}",
            f"--port={db_params['port']}",
            "--command=DROP TABLE IF EXISTS test_restore;"
        ]
        subprocess.run(drop_cmd, env=env, capture_output=True) 