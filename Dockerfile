FROM python:3.8-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Установка переменных окружения для GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data/satellite_images data/seismic checkpoints logs

# Создание скрипта инициализации
RUN echo '#!/bin/bash\n\
echo "Waiting for PostgreSQL..."\n\
while ! pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; do\n\
    sleep 1\n\
done\n\
echo "PostgreSQL started"\n\
python init_db.py\n\
python run_collector.py' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Создание healthcheck.py
RUN echo 'import psycopg2\n\
import os\n\
import sys\n\
\n\
def check_db():\n\
    try:\n\
        conn = psycopg2.connect(\n\
            host=os.getenv("DB_HOST"),\n\
            port=os.getenv("DB_PORT"),\n\
            dbname=os.getenv("DB_NAME"),\n\
            user=os.getenv("DB_USER"),\n\
            password=os.getenv("DB_PASSWORD")\n\
        )\n\
        conn.close()\n\
        return True\n\
    except:\n\
        return False\n\
\n\
if __name__ == "__main__":\n\
    if check_db():\n\
        sys.exit(0)\n\
    sys.exit(1)' > /app/healthcheck.py \
    && chmod +x /app/healthcheck.py

# Настройка прав доступа
RUN chown -R nobody:nogroup /app
USER nobody

# Запуск приложения
CMD ["/app/entrypoint.sh"] 