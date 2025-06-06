import os
from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection, MimeType, BBox, CRS, bbox_to_dimensions
)
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
from affine import Affine
import math
import json

class SentinelHubOAuthClient:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        # Координаты вулкана Ключевской (можно менять)
        self.volcano_bbox = [160.45, 55.95, 160.65, 56.10]
        self.resolution = 15
        self.max_cloud_cover = 20
        self.max_image_size = 2500
        self.evalscript = self._get_evalscript()

    def _load_config(self, config_path=None):
        config = SHConfig()
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                creds = json.load(f)
                config.sh_client_id = creds['client_id']
                config.sh_client_secret = creds['client_secret']
        else:
            config.sh_client_id = os.getenv('SENTINEL_CLIENT_ID')
            config.sh_client_secret = os.getenv('SENTINEL_CLIENT_SECRET')
        if not config.sh_client_id or not config.sh_client_secret:
            raise ValueError("Необходимо указать client_id и client_secret Sentinel Hub")
        return config

    def _get_evalscript(self):
        return """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B03", "B02", "CLM"],
                output: { bands: 4, sampleType: "FLOAT32" }
            };
        }
        function evaluatePixel(sample) {
            return [
                2.5 * sample.B04,
                2.5 * sample.B03,
                2.5 * sample.B02,
                sample.CLM
            ];
        }
        """

    def download_image(self, start_date, end_date, output_dir: Path, bbox=None, resolution=None, max_cloud_cover=None):
        bbox = bbox or self.volcano_bbox
        resolution = resolution or self.resolution
        max_cloud_cover = max_cloud_cover or self.max_cloud_cover
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bbox_obj = BBox(bbox=bbox, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox_obj, resolution=resolution)
        if size[0] > self.max_image_size or size[1] > self.max_image_size:
            scale_factor = max(size[0] / self.max_image_size, size[1] / self.max_image_size)
            size = (
                math.ceil(size[0] / scale_factor),
                math.ceil(size[1] / scale_factor)
            )
        request = SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    maxcc=max_cloud_cover / 100,
                    mosaicking_order='leastCC'
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_obj,
            size=size,
            config=self.config
        )
        data = request.get_data()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = output_dir / f"Klyuchevskoy_{start_date}_{end_date}_{timestamp}.tiff"
        transform = Affine.from_gdal(*bbox_obj.get_transform_vector(size[0], size[1]))
        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            width=size[0],
            height=size[1],
            count=4,
            dtype=np.float32,
            crs=CRS.WGS84.pyproj_crs(),
            transform=transform
        ) as dst:
            if data[0].shape != (size[1], size[0], 4):
                raise ValueError("Несоответствие размеров данных и метаданных")
            for i in range(4):
                dst.write(data[0][:, :, i], i + 1)
            dst.update_tags(
                BBOX=str(bbox),
                DATE=datetime.now().isoformat(),
                CLOUD_COVER=max_cloud_cover
            )
        return filename 