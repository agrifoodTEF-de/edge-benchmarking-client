#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest

from pathlib import Path
from dotenv import load_dotenv
from edge_benchmarking_client.client import EdgeBenchmarkingClient
from edge_benchmarking_types.edge_farm.models import TritonInferenceClientConfig

# Connection information
PROTOCOL = "https"
HOST = "api.edge-farm.agrifood-tef.edvsz.hs-osnabrueck.de"
EDGE_DEVICE_HOST = "edge-03"

EXAMPLES_ROOT_DIR = Path("../examples").expanduser().resolve()
EXAMPLE_ROOT_DIR = "densenet_onnx"


class TestEdgeBenchmarkingClient:
    def setup_class(self) -> None:
        load_dotenv(dotenv_path=EXAMPLES_ROOT_DIR.joinpath(".env"))

        # Basic API authentication
        BASIC_AUTH_USERNAME = os.getenv("BASIC_AUTH_USERNAME")
        BASIC_AUTH_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD")

        self.client = EdgeBenchmarkingClient(
            protocol=PROTOCOL,
            host=HOST,
            username=BASIC_AUTH_USERNAME,
            password=BASIC_AUTH_PASSWORD,
        )

        self.inference_client_config = TritonInferenceClientConfig(
            host=EDGE_DEVICE_HOST,
            model_name=EXAMPLE_ROOT_DIR,
            num_classes=10,
            scaling="inception",
        )

    def test_get_device_header(self) -> None:
        device_header = self.client.get_device_header(hostname=EDGE_DEVICE_HOST)
        assert device_header.hostname == EDGE_DEVICE_HOST
        assert device_header.online

    def test_get_device_info(self) -> None:
        device_info = self.client.get_device_info(hostname=EDGE_DEVICE_HOST)
        assert device_info.platform.system == "Linux"

    def test_find_dataset(self) -> None:
        # TODO:
        pass

    def test_find_model(self) -> None:
        # TODO:
        pass

    def test_find_model_metadata(self) -> None:
        # TODO:
        pass

    def test_find_labels(self) -> None:
        # TODO:
        pass

    def test_benchmark(self) -> None:
        # TOOD:
        pass


if __name__ == "__main__":
    pytest.main()
