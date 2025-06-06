-- Миграция базы данных
-- Описание: Создание основных таблиц

BEGIN;

-- Таблица для спутниковых данных
CREATE TABLE IF NOT EXISTS satellite_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    ndvi FLOAT,
    ndbi FLOAT,
    ndbai FLOAT,
    thermal_anomaly BOOLEAN,
    smoke_detected BOOLEAN,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица для сейсмических данных
CREATE TABLE IF NOT EXISTS seismic_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    magnitude FLOAT,
    depth FLOAT,
    latitude FLOAT,
    longitude FLOAT,
    event_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица для извержений
CREATE TABLE IF NOT EXISTS eruptions (
    id SERIAL PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    magnitude FLOAT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица для предсказаний
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    probability FLOAT NOT NULL,
    threshold FLOAT NOT NULL,
    is_eruption_predicted BOOLEAN NOT NULL,
    feature_importance JSONB,
    temporal_importance JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Создание индексов
CREATE INDEX IF NOT EXISTS idx_satellite_timestamp 
ON satellite_data(timestamp);

CREATE INDEX IF NOT EXISTS idx_seismic_timestamp 
ON seismic_data(timestamp);

CREATE INDEX IF NOT EXISTS idx_eruptions_start_time 
ON eruptions(start_time);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
ON predictions(timestamp);

COMMIT; 