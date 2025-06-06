from satellite_data_collector import SatelliteDataCollector
from pathlib import Path

if __name__ == '__main__':
    data_dir = Path('data/period_collection/optical')
    collector = SatelliteDataCollector(processed_data_dir=str(data_dir))
    print(f'Обработка всех TIFF-файлов в {data_dir}...')
    results = collector.process_test_images(str(data_dir))
    print('\nРезультаты обработки:')
    print(f"Всего снимков: {results.get('total_images', 0)}")
    print(f"Обработано успешно: {len(results.get('processed_images', []) )}")
    print(f"Отклонено: {len(results.get('rejected_images', []) )}")
    if results.get('rejected_images'):
        print('\nПричины отклонения снимков:')
        for img in results['rejected_images']:
            print(f"\nФайл: {img['filename']}")
            print(f"Облачность: {img['cloud_coverage']:.2%}")
            if img.get('acquisition_time'):
                print(f"Время съемки: {img['acquisition_time']}")
    print('\nПроверьте PNG-превью в папке:', data_dir) 