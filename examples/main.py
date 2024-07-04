#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

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
    labels = client.find_labels(root_dir=EXAMPLE_ROOT_DIR)

    # Start benchmark
    benchmark_results, inference_results = client.benchmark(
        edge_device="edge-03",
        dataset=dataset,
        model=model,
        model_name=EXAMPLE_ROOT_DIR,
        model_metadata=model_metadata,
        labels=labels,
        num_classes=1000,
    )

    # Benchmark results
    times = benchmark_results["time"]  # x axis
    gpu = benchmark_results["GPU"]
    ram = benchmark_results["RAM"]
    cpu_temp = benchmark_results["Temp CPU"]

    print(times, gpu, ram, cpu_temp)

    # Inference results
    inference_results_table = np.stack(inference_results)

    for sample in inference_results_table:
        logits = sample[:, 0].astype(float)
        predicted_classes = sample[:, -1]
        predicted_class = predicted_classes[logits.argmax()]
        print(predicted_class)
