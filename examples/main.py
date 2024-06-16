#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv
from edge_benchmarking_client.client import EdgeBenchmarkingClient

if __name__ == "__main__":
    load_dotenv()

    # Connection information
    PROTOCOL = "https"
    HOST = "api.edge-farm.agrifood-tef.edvsz.hs-osnabrueck.de"

    # Basic API authentication
    BASIC_AUTH_USERNAME = os.getenv("BASIC_AUTH_USERNAME")
    BASIC_AUTH_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD")

    # Create the client
    client = EdgeBenchmarkingClient(
        protocol=PROTOCOL,
        host=HOST,
        username=BASIC_AUTH_USERNAME,
        password=BASIC_AUTH_PASSWORD,
    )

    # Infer benchmarking job components: (dataset, model, model_metadata)
    EXAMPLE_ROOT_DIR = "densenet_onnx"

    dataset = client.find_dataset(
        root_dir=EXAMPLE_ROOT_DIR, file_extensions={".jpg", ".jpeg"}
    )
    model = client.find_model(root_dir=EXAMPLE_ROOT_DIR)
    model_metadata = client.find_model_metadata(root_dir=EXAMPLE_ROOT_DIR)

    # Start benchmark
    benchmark_results = client.benchmark(
        dataset=dataset, model=model, model_metadata=model_metadata
    )

    # Use matplotlib to visualize 'benchmark_results'
    # TODO
