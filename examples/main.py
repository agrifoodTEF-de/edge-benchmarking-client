#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from dotenv import load_dotenv
from collections import defaultdict
from edge_benchmarking_client.client import EdgeBenchmarkingClient
from edge_benchmarking_types.edge_farm.models import TritonInferenceClientConfig

if __name__ == "__main__":
    load_dotenv()

    # Connection information
    PROTOCOL = "https"
    HOST = "api.edge-farm.agrifood-tef.edvsz.hs-osnabrueck.de"
    EDGE_DEVICE_HOST = "edge-03"

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

    # Fetch device header and info
    device_header = client.get_device_header(hostname=EDGE_DEVICE_HOST)
    print(device_header)

    device_info = client.get_device_info(hostname=EDGE_DEVICE_HOST)
    print(device_info)

    # Infer benchmarking job components: (dataset, model, model_metadata)
    EXAMPLE_ROOT_DIR = "densenet_onnx"

    dataset = client.find_dataset(
        root_dir=EXAMPLE_ROOT_DIR, file_extensions={".jpg", ".jpeg"}
    )
    model = client.find_model(root_dir=EXAMPLE_ROOT_DIR)
    model_metadata = client.find_model_metadata(root_dir=EXAMPLE_ROOT_DIR)
    labels = client.find_labels(root_dir=EXAMPLE_ROOT_DIR)

    # Create inference client configuration (in this case for Triton)
    inference_client_config = TritonInferenceClientConfig(
        host=EDGE_DEVICE_HOST,
        model_name=EXAMPLE_ROOT_DIR,
        num_classes=10,
        scaling="inception",
    )

    # Start benchmark
    benchmark_results, inference_results = client.benchmark(
        edge_device=EDGE_DEVICE_HOST,
        dataset=dataset,
        model=model,
        model_metadata=model_metadata,
        labels=labels,
        inference_client_config=inference_client_config,
    )

    # Benchmark results
    benchmark_results = pd.DataFrame(benchmark_results)
    print(benchmark_results)

    # Inference results
    final_inference_results = defaultdict(list)
    for inference_respone_id, inference_result in inference_results.items():
        predictions = np.stack(inference_result)

        logits = predictions[:, 0].astype(float)
        probabilities = F.softmax(torch.tensor(logits), dim=0)

        predicted_classes = predictions[:, -1]
        predicted_class_index = probabilities.argmax()
        predicted_probability = probabilities.max()
        predicted_class = predicted_classes[predicted_class_index]

        final_inference_results["response id"].append(inference_respone_id)
        final_inference_results["class"].append(predicted_class)
        final_inference_results["probability"].append(
            predicted_probability.item() * 100
        )

        inference_results_df = pd.DataFrame(final_inference_results)
    print(inference_results_df)
