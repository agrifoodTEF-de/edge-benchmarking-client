#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from edge_benchmarking_types.edge_device.enums import JobStatus
from edge_benchmarking_client.client import EdgeBenchmarkingClient
from edge_benchmarking_types.edge_farm.enums import OakImageResolution
from edge_benchmarking_types.edge_farm.models import (
    TritonDenseNetClient,
    ExternalDataProvider,
    OakClient,
)

if __name__ == "__main__":
    load_dotenv()

    EDGE_DEVICE_HOST = "edge-03"

    # Basic API authentication
    EDGE_FARM_API_BASIC_AUTH_USERNAME = os.getenv("EDGE_FARM_API_BASIC_AUTH_USERNAME")
    EDGE_FARM_API_BASIC_AUTH_PASSWORD = os.getenv("EDGE_FARM_API_BASIC_AUTH_PASSWORD")

    # Connection information
    EDGE_FARM_API_PROTOCOL = os.getenv("EDGE_FARM_API_PROTOCOL")
    EDGE_FARM_API_HOST = os.getenv("EDGE_FARM_API_HOST")

    # Create the client
    client = EdgeBenchmarkingClient(
        protocol=EDGE_FARM_API_PROTOCOL,
        host=EDGE_FARM_API_HOST,
        username=EDGE_FARM_API_BASIC_AUTH_USERNAME,
        password=EDGE_FARM_API_BASIC_AUTH_PASSWORD,
    )

    # Fetch device header and info
    device_header = client.get_device_header(hostname=EDGE_DEVICE_HOST)
    print(device_header)

    device_info = client.get_device_info(hostname=EDGE_DEVICE_HOST)
    print(device_info)

    # Upload a dataset from disk or use external sensor (camera, etc.)?
    USE_LOCAL_DATASET = True

    # Infer benchmarking job components: (dataset, model, model_metadata)
    EXAMPLE_ROOT_DIR = "densenet_onnx"

    if USE_LOCAL_DATASET:
        dataset = client.find_dataset(
            root_dir=EXAMPLE_ROOT_DIR,
            # file_extensions={".jpg", ".jpeg", ".JPEG", ".JPG", ".png", ".PNG"},
            file_extensions={".JPEG"},
        )
    else:
        OAK_CAMERA_IP = "192.168.1.100"
        OAK_MAX_SAMPLE_SIZE = 10
        CAPTURE_ROOT_DIR = "capture"

        oak_client = OakClient(
            ip=OAK_CAMERA_IP, rgb_resolution=OakImageResolution.THE_1080P, warmup=3
        )
        external_data_provider = ExternalDataProvider(
            client=oak_client,
            max_sample_size=OAK_MAX_SAMPLE_SIZE,
        )
        dataset = client.capture_dataset(
            root_dir=CAPTURE_ROOT_DIR, external_data_provider=external_data_provider
        )

    model = client.find_model(root_dir=EXAMPLE_ROOT_DIR)
    model_metadata = client.find_model_metadata(root_dir=EXAMPLE_ROOT_DIR)
    labels = client.find_labels(root_dir=EXAMPLE_ROOT_DIR)

    # Create inference client configuration (in this case for Triton with DenseNet model)
    inference_client = TritonDenseNetClient(
        protocol="http",
        host=EDGE_DEVICE_HOST,
        port=8000,
        num_workers=1,
        samples_per_second=10,
        warm_up=False,
        model_name=EXAMPLE_ROOT_DIR,
        model_version="1",
        batch_size=1,
        num_classes=1000,
        scaling="inception",
    )

    # Start benchmark
    benchmark_job = client.benchmark(
        edge_device=EDGE_DEVICE_HOST,
        dataset=dataset,
        model=model,
        model_metadata=model_metadata,
        labels=labels,
        inference_client=inference_client,
        chunk_size=10,
        cpu_only=False,
        cleanup=True,
    )

    # If benchmark job has failed, read error message
    if benchmark_job.status == JobStatus.FAILED:
        print("Benchmark job has failed:", benchmark_job.error)
    else:
        # Benchmark results
        benchmark_results = pd.DataFrame(benchmark_job.benchmark_results)
        print(benchmark_results)

        # Inference results
        final_inference_results = defaultdict(list)
        for (
            inference_respone_id,
            inference_result,
        ) in benchmark_job.inference_results.results.items():
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
