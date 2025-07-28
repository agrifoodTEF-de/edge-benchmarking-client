#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import pytest

from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from requests import codes, HTTPError
from edge_benchmarking_client.client import EdgeBenchmarkingClient
from edge_benchmarking_types.sensors.models import Sensor, OakClient
from edge_benchmarking_types.sensors.enums import OakImageResolution
from edge_benchmarking_types.edge_farm.models import TritonDenseNetClient


EDGE_DEVICE_HOST = "edge-03"
# EDGE_DEVICE_HOST = "edge-09"

EXAMPLES_ROOT_DIR = Path("examples")
DENSENET_ROOT_DIR = "densenet_onnx"
# DENSENET_ROOT_DIR = "yolov11_onnx"
CAPTURE_ROOT_DIR = EXAMPLES_ROOT_DIR.joinpath("capture")

SENSOR_HOSTNAME_NOT_EXISTING = "does-not-exist"
OAK_CAMERA_HOSTNAME = "cam-01"
OAK_CAMERA_IP = "192.168.1.100"
OAK_MAX_SAMPLE_SIZE = 10


class TestEdgeBenchmarkingClient:
    def setup_class(self) -> None:
        load_dotenv(dotenv_path=EXAMPLES_ROOT_DIR.joinpath(".env"))

        # Basic API authentication
        EDGE_FARM_API_BASIC_AUTH_USERNAME = os.getenv(
            "EDGE_FARM_API_BASIC_AUTH_USERNAME"
        )
        EDGE_FARM_API_BASIC_AUTH_PASSWORD = os.getenv(
            "EDGE_FARM_API_BASIC_AUTH_PASSWORD"
        )

        # Connection information
        EDGE_FARM_API_PROTOCOL = os.getenv("EDGE_FARM_API_PROTOCOL")
        EDGE_FARM_API_HOST = os.getenv("EDGE_FARM_API_HOST")

        self.client = EdgeBenchmarkingClient(
            protocol=EDGE_FARM_API_PROTOCOL,
            host=EDGE_FARM_API_HOST,
            username=EDGE_FARM_API_BASIC_AUTH_USERNAME,
            password=EDGE_FARM_API_BASIC_AUTH_PASSWORD,
        )

    def test_get_welcome_message(self) -> None:
        welcome_message = self.client.get_welcome_message().json()
        assert len(welcome_message) == 1
        assert "message" in welcome_message

    def test_get_device_header(self) -> None:
        device_header = self.client.get_device_header(hostname=EDGE_DEVICE_HOST)
        assert device_header.hostname == EDGE_DEVICE_HOST
        assert device_header.online

    def test_get_sensors(self) -> None:
        sensors = self.client.get_sensors()
        assert len(sensors)
        assert any(sensor.hostname == OAK_CAMERA_HOSTNAME for sensor in sensors)

    def test_get_sensor(self) -> None:
        sensor = self.client.get_sensor(hostname=OAK_CAMERA_HOSTNAME)
        assert sensor.hostname == OAK_CAMERA_HOSTNAME

    def test_get_sensor_not_found(self) -> None:
        try:
            self.client.get_sensor(hostname=SENSOR_HOSTNAME_NOT_EXISTING)
        except HTTPError as e:
            assert e.response.status_code == codes.not_found

    def test_get_device_info(self) -> None:
        device_info = self.client.get_device_info(hostname=EDGE_DEVICE_HOST)
        assert device_info.platform.system == "Linux"

    def test_find_dataset(self) -> None:
        file_extensions = {".JPEG"}
        dataset = self.client.find_dataset(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR),
            file_extensions=file_extensions,
        )
        assert len(dataset) == 50
        assert {sample.suffix for sample in dataset} <= file_extensions
        assert all("dataset" == sample.parent.name for sample in dataset)

    def test_find_model(self) -> None:
        model = self.client.find_model(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
        )
        assert isinstance(model, Path)
        assert model.suffix == ".onnx"
        assert model.parent.name == "model"

    def test_find_model_metadata(self) -> None:
        model_metadata = self.client.find_model_metadata(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
        )
        assert isinstance(model_metadata, Path)
        assert model_metadata.suffix == ".pbtxt"
        assert model_metadata.parent.name == "model"

    def test_find_labels(self) -> None:
        labels = self.client.find_labels(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
        )
        assert isinstance(labels, Path)
        assert labels.suffix == ".txt"
        assert labels.parent.name == "model"

    def test_benchmark_dataset_jpeg(self) -> None:
        dataset = self.client.find_dataset(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR),
            file_extensions={".JPEG"},
        )
        self._benchmark_files_dataset(dataset)
        self._benchmark_bytes_dataset(dataset)

    def test_benchmark_dataset_archives(self) -> None:
        dataset = self.client.find_dataset(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR),
            file_extensions={".zip", ".tar"},
        )
        self._benchmark_files_dataset(dataset)
        self._benchmark_bytes_dataset(dataset)

    def test_benchmark_sensor_capture_oak(self) -> None:
        oak_client = OakClient(
            ip=OAK_CAMERA_IP, rgb_resolution=OakImageResolution.THE_1080P, warmup=3
        )
        sensor = Sensor(client=oak_client, max_sample_size=OAK_MAX_SAMPLE_SIZE)
        dataset = self.client.capture_dataset(root_dir=CAPTURE_ROOT_DIR, sensor=sensor)
        self._benchmark_files_dataset(dataset)
        self._benchmark_bytes_dataset(dataset)

    def _benchmark_files_dataset(self, dataset: list[Path]) -> None:
        model = self.client.find_model(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
        )
        model_metadata = self.client.find_model_metadata(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
        )
        labels = self.client.find_labels(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
        )

        self._test_benchmark(
            dataset=dataset,
            model=model,
            model_metadata=model_metadata,
            labels=labels,
            chunk_size=10,
        )

    def _benchmark_bytes_dataset(self, dataset: list[Path]) -> None:
        files = {
            "model": self.client.find_model(
                root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
            ),
            "model_metadata": self.client.find_model_metadata(
                root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
            ),
            "labels": self.client.find_labels(
                root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR)
            ),
        }

        for name, filepath in files.items():
            with open(filepath, "rb") as fh:
                files[name] = (filepath.name, BytesIO(fh.read()))

        files["dataset"] = []
        for sample in dataset:
            with open(sample, "rb") as fh:
                files["dataset"].append((sample.name, BytesIO(fh.read())))

        self._test_benchmark(
            dataset=files["dataset"],
            model=files["model"],
            model_metadata=files["model_metadata"],
            labels=files["labels"],
            cpu_only=False,
        )

    def _test_benchmark(
        self,
        dataset: Path | tuple[str, BytesIO],
        model: Path | tuple[str, BytesIO],
        model_metadata: Path | tuple[str, BytesIO],
        labels: Path | tuple[str, BytesIO],
        cleanup: bool = True,
        chunk_size: int | None = None,
        protocol: str = "http",
        port: int | None = 8000,
        batch_size: int = 1,
        cpu_only: bool = False,
        num_workers: int = 1,
        samples_per_second: float | None = 10,
        warm_up: bool = False,
        num_classes: int = 10,
        scaling: str | None = "inception",
    ) -> None:
        inference_client = TritonDenseNetClient(
            protocol=protocol,
            host=EDGE_DEVICE_HOST,
            port=port,
            num_workers=num_workers,
            samples_per_second=samples_per_second,
            warm_up=warm_up,
            model_name=DENSENET_ROOT_DIR,
            model_version="1",
            batch_size=batch_size,
            num_classes=num_classes,
            scaling=scaling,
        )

        benchmark_job = self.client.benchmark(
            edge_device=EDGE_DEVICE_HOST,
            dataset=dataset,
            model=model,
            model_metadata=model_metadata,
            labels=labels,
            inference_client=inference_client,
            cleanup=cleanup,
            chunk_size=chunk_size,
            cpu_only=cpu_only,
        )

        assert all(
            len(predictions) == num_classes
            for predictions in benchmark_job.inference_results.results.values()
        )
        assert {"time", "CPU1", "GPU", "RAM", "Temp CPU", "Temp GPU"}.issubset(
            benchmark_job.benchmark_results.keys()
        )


if __name__ == "__main__":
    pytest.main()
