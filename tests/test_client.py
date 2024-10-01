#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest

from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from edge_benchmarking_client.client import EdgeBenchmarkingClient
from edge_benchmarking_types.edge_farm.models import TritonInferenceClientConfig

EDGE_DEVICE_HOST = "edge-03"

EXAMPLES_ROOT_DIR = Path("examples")
DENSENET_ROOT_DIR = "densenet_onnx"


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

    def test_get_device_info(self) -> None:
        device_info = self.client.get_device_info(hostname=EDGE_DEVICE_HOST)
        assert device_info.platform.system == "Linux"

    def test_find_dataset(self) -> None:
        file_extensions = {".jpg", ".jpeg"}
        dataset = self.client.find_dataset(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR),
            file_extensions=file_extensions,
        )
        assert len(dataset) == 3
        assert {sample.suffix for sample in dataset} <= file_extensions
        assert all("base_dataset" == sample.parent.name for sample in dataset)

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

    def test_benchmark_files(self) -> None:
        dataset = self.client.find_dataset(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR),
            file_extensions={".jpg", ".jpeg"},
        )
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
            dataset=dataset, model=model, model_metadata=model_metadata, labels=labels
        )

    def test_benchmark_bytes(self) -> None:
        dataset = self.client.find_dataset(
            root_dir=EXAMPLES_ROOT_DIR.joinpath(DENSENET_ROOT_DIR),
            file_extensions={".jpg", ".jpeg"},
        )

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
        )

    def _test_benchmark(
        self,
        dataset: Path | tuple[str, BytesIO],
        model: Path | tuple[str, BytesIO],
        model_metadata: Path | tuple[str, BytesIO],
        labels: Path | tuple[str, BytesIO],
        cleanup: bool = True,
    ) -> None:
        num_classes = 10
        inference_client_config = TritonInferenceClientConfig(
            host=EDGE_DEVICE_HOST,
            model_name=DENSENET_ROOT_DIR,
            num_classes=num_classes,
            scaling="inception",
        )

        benchmark_job = self.client.benchmark(
            edge_device=EDGE_DEVICE_HOST,
            dataset=dataset,
            model=model,
            model_metadata=model_metadata,
            labels=labels,
            inference_client_config=inference_client_config,
            cleanup=cleanup,
        )

        assert all(
            len(predictions) == num_classes
            for predictions in benchmark_job.inference_results.values()
        )
        assert {"time", "CPU1", "GPU", "RAM", "Temp CPU", "Temp GPU"}.issubset(
            benchmark_job.benchmark_results.keys()
        )


if __name__ == "__main__":
    pytest.main()
