import pandas as pd
import numpy as np
from pathlib import Path

def load_events(file_path):
    """
    Загружает и очищает данные о событиях (землетрясения, извержения) из CSV или Excel.
    Возвращает pandas DataFrame с приведёнными типами.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', na_values=['', '', ' '])
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, sep=';', encoding='cp1251', na_values=['', '', ' '])
    elif file_path.suffix.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path, na_values=['', '', ' '])
    else:
        raise ValueError('Неподдерживаемый формат файла')

    # Приводим даты и время к единому формату
    if np.issubdtype(df['Date'].dtype, np.datetime64):
        # Если дата уже datetime, просто объединяем с временем
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce', dayfirst=True)
    else:
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=True)
    # Преобразуем координаты и глубину
    for col in ['Latitude', 'Longitude', 'Depth (km)', 'Ml', 'Mc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Оставляем только события с валидными координатами и датой
    df = df.dropna(subset=['datetime', 'Latitude', 'Longitude'])
    # Сортируем по времени
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def filter_events_by_periods(events_df, periods):
    """
    Фильтрует события по списку периодов.
    Args:
        events_df: DataFrame с землетрясениями (должен содержать столбец 'datetime')
        periods: список кортежей (start_date, end_date)
    Returns:
        Список списков: для каждого периода — DataFrame с событиями в этом периоде
    """
    filtered = []
    for start, end in periods:
        mask = (events_df['datetime'] >= start) & (events_df['datetime'] < end)
        filtered.append(events_df[mask].copy())
    return filtered

if __name__ == "__main__":
    # Пример использования
    for fname in ["Test.csv", "Test.xlsx"]:
        try:
            events = load_events(fname)
            print(f"\n{fname}: {len(events)} событий")
            print(events.head())
        except Exception as e:
            print(f"Ошибка при обработке {fname}: {e}") 