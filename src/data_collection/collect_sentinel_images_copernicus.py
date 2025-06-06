import requests
from pathlib import Path

def collect_sentinel_images_copernicus(data_dir: str = "data/raw"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print("Выполняем аутентификацию...")
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    auth_data = {
        "client_id": "cdse-public",
        "username": "katekeo000@gmail.com",
        "password": "Kate11041506$",
        "grant_type": "password"
    }
    try:
        auth_response = requests.post(auth_url, data=auth_data)
        auth_response.raise_for_status()
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("Аутентификация успешна!")
    except Exception as e:
        print(f"Ошибка при аутентификации: {e}")
        return
    # AOI POLYGON (должен начинаться и заканчиваться одной и той же точкой)
    aoi = "POLYGON((158.0 55.0,162.0 55.0,162.0 57.0,158.0 57.0,158.0 55.0))"
    filter_str = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') "
    )
    params = {
        "$filter": filter_str,
        "$top": 1
    }
    print(f"Фильтр: {filter_str}")
    search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        products = response.json().get("value", [])
        if not products:
            print(f"Снимки не найдены!")
            return
        print(f"Найдено снимков: {len(products)}")
        product = products[0]
        file_name = f"Klyuchevskoy_{product['Name']}.zip"
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"Файл {file_name} уже существует")
            return
        print(f"Найден снимок: {product['Name']}")
        print(f"Дата: {product['ContentDate']['Start']}\nРазмер: {product.get('Size', 'N/A')}")
        download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value"
        download_file(download_url, headers, file_path)
    except Exception as e:
        print(f"Ошибка поиска: {e}")

def download_file(url: str, headers: dict, file_path: Path, chunk_size: int = 10 * 1024 * 1024):
    try:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(file_path, 'wb') as f:
                if total_size == 0:
                    f.write(r.content)
                else:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = downloaded / total_size * 100
                            print(f"\rСкачивание: {progress:.1f}% ({downloaded/1024/1024:.1f} MB)", end='')
        print(f"\nФайл сохранен: {file_path}")
        return True
    except Exception as e:
        print(f"\nОшибка скачивания: {e}")
        if file_path.exists():
            file_path.unlink()
        return False

if __name__ == "__main__":
    collect_sentinel_images_copernicus() 