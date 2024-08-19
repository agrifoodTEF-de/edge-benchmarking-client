import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import os

import math
import time
import urllib
import requests
import validators

from io import BytesIO
from pathlib import Path
from requests import Response
from typing import Any, Union
from requests.auth import HTTPBasicAuth
from edge_benchmarking_client.endpoints import (
    DEVICE,
    BENCHMARK_DATA,
    BENCHMARK_JOB,
)
from edge_benchmarking_types.edge_farm.models import (
    BenchmarkData,
    EdgeDeviceConfig,
    TritonInferenceClientConfig,
)
from edge_benchmarking_types.edge_device.enums import JobStatus
from edge_benchmarking_types.edge_device.models import (
    DeviceInfo,
    DeviceHeader,
    BenchmarkJob,
)

SUPPORTED_MODEL_FORMATS = {".onnx", ".pt", ".pth"}


class EdgeBenchmarkingClient:
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        port: int | None = None,
        protocol: str = "https",
    ) -> None:
        self.protocol = protocol
        self.host = host
        self.port = port
        self.username = username
        self.auth = HTTPBasicAuth(username, password)

        self.api = f"{protocol}://{host}"
        if port is not None:
            self.api += f":{port}"

        if not validators.url(self.api):
            raise ValueError(f"Invalid Edge Farm API URL '{self.api}'.")

        logging.info(
            f"Created Edge Benchmarking client for user '{username}' and Edge Farm at '{self.api}'."
        )
        self._test_connection()

    def _test_connection(self) -> None:
        response = self.get_welcome_message()
        logging.info(
            f"Edge Farm API at '{self.api}' is reachable with status code {response.status_code}."
        )

    @staticmethod
    def _collect_files(
        root_dir: str, file_extensions: set[str]
    ) -> tuple[list[Path], Path]:
        filepaths = []
        root_path = Path(root_dir).expanduser().resolve()
        for ext in file_extensions:
            filepaths.extend(root_path.rglob(f"*{ext}"))
        return filepaths, root_path

    def _find_file(
        self, root_dir: str, extensions: set[str], filename: str | None = None
    ) -> Path:
        filepaths, root_path = self._collect_files(
            root_dir=root_dir, file_extensions=extensions
        )
        if filename is not None:
            filepaths = [path for path in filepaths if path.name == filename]

        if not filepaths:
            raise RuntimeError(
                f"No file with extension {extensions} found in '{root_path}'."
            )

        if len(filepaths) > 1:
            logging.warning(
                f"Found multiple ({len(filepaths)}) files with extension {extensions} in '{root_path}'. Selecting first match."
            )

        filepath = filepaths[0]
        logging.info(
            f"Found file '{filepath}' with extension {extensions} in '{root_path}'."
        )
        return filepath

    def _endpoint(self, *paths, query: dict | None = None) -> str:
        url = self.api + os.path.join(*paths)
        if query is not None:
            query_string = urllib.parse.urlencode(query)
            url += f"?{query_string}"
        if not validators.url(url):
            raise RuntimeError(f"Invalid URL: {url}")
        return url

    def get_welcome_message(self) -> Response:
        response = requests.get(self.api, auth=self.auth)
        response.raise_for_status()
        logging.info(f"{response.status_code} - {response.json()}")
        return response

    def find_dataset(self, root_dir: str, file_extensions: set[str]) -> list[Path]:
        sample_filepaths, root_path = self._collect_files(
            root_dir=root_dir, file_extensions=file_extensions
        )
        logging.info(
            f"Found dataset containing {len(sample_filepaths)} samples with type(s) {file_extensions} in '{root_path}'."
        )
        return sample_filepaths

    def find_model(self, root_dir: str, model_name: str | None = None) -> Path:
        return self._find_file(
            root_dir=root_dir, extensions=SUPPORTED_MODEL_FORMATS, filename=model_name
        )

    def find_model_metadata(
        self, root_dir: str, model_metadata_name: str | None = None
    ) -> Path:
        return self._find_file(
            root_dir=root_dir, extensions={".pbtxt"}, filename=model_metadata_name
        )

    def find_labels(self, root_dir: str, labels_name: str | None = None) -> Path:
        return self._find_file(
            root_dir=root_dir, extensions={".txt"}, filename=labels_name
        )

    def upload_benchmark_data(
        self,
        dataset: list[Path] | list[tuple[str, BytesIO]],
        model: Path | tuple[str, BytesIO],
        model_metadata: Path | tuple[str, BytesIO] | None = None,
        labels: Path | tuple[str, BytesIO] | None = None,
    ) -> BenchmarkData:
        benchmark_data_files = []
        try:
            assert len(dataset), "Dataset is empty."
            dataset_data = [
                (
                    "dataset",
                    (
                        sample
                        if isinstance(sample, tuple)
                        else (sample.name, open(sample, "rb"))
                    ),
                )
                for sample in dataset
            ]
            model_data = (
                ("model", model)
                if isinstance(model, tuple)
                else ("model", (model.name, open(model, "rb")))
            )

            benchmark_data_files = dataset_data + [model_data]

            if model_metadata is not None:
                model_metadata = (
                    ("model_metadata", model_metadata)
                    if isinstance(model_metadata, tuple)
                    else (
                        "model_metadata",
                        (model_metadata.name, open(model_metadata, "rb")),
                    )
                )
                benchmark_data_files.append(model_metadata)

            if labels is not None:
                labels_data = (
                    ("labels", labels)
                    if isinstance(labels, tuple)
                    else ("labels", (labels.name, open(labels, "rb")))
                )
                benchmark_data_files.append(labels_data)

            response = requests.post(
                url=self._endpoint(BENCHMARK_DATA),
                files=benchmark_data_files,
                auth=self.auth,
            )
            response.raise_for_status()
            benchmark_data = BenchmarkData.model_validate(response.json())
            logging.info(f"{response.status_code} - {benchmark_data}")
            return benchmark_data
        finally:
            for _, (_, payload) in benchmark_data_files:
                payload.close()

    def start_benchmark_job(
        self,
        job_id: str,
        edge_device_config: EdgeDeviceConfig,
        inference_client_config: Union[TritonInferenceClientConfig],
    ) -> Response:
        response = requests.post(
            url=self._endpoint(BENCHMARK_JOB, job_id, "start"),
            json={
                "edge_device": edge_device_config.model_dump(),
                "inference_client": inference_client_config.model_dump(),
            },
            auth=self.auth,
        )
        response.raise_for_status()
        logging.info(f"{response.status_code}")
        return response

    def get_benchmark_job(self, job_id: str) -> BenchmarkJob:
        response = requests.get(
            url=self._endpoint(BENCHMARK_JOB, job_id), auth=self.auth
        )
        response.raise_for_status()
        benchmark_job = BenchmarkJob.model_validate(response.json())
        logging.info(f"{response.status_code} - {benchmark_job}")
        return benchmark_job

    def get_benchmark_job_status(self, job_id: str) -> dict[str, JobStatus]:
        response = requests.get(
            url=self._endpoint(BENCHMARK_JOB, job_id, "status"), auth=self.auth
        )
        response.raise_for_status()
        job_status = response.json()
        logging.info(f"{response.status_code} - {job_status}")
        return job_status

    def get_device_header(self, hostname: str) -> DeviceHeader:
        response = requests.get(
            url=self._endpoint(DEVICE, hostname, "header"), auth=self.auth
        )
        response.raise_for_status()
        device_header = DeviceHeader.model_validate(response.json())
        logging.info(f"{response.status_code} - {device_header}")
        return device_header

    def get_device_info(self, hostname: str) -> DeviceInfo:
        response = requests.get(
            url=self._endpoint(DEVICE, hostname, "info"), auth=self.auth
        )
        response.raise_for_status()
        device_info = DeviceInfo.model_validate(response.json())
        logging.info(f"{response.status_code} - {device_info}")
        return device_info

    def get_benchmark_job_results(
        self, job_id: str, max_retries: int = math.inf, patience: int = 1
    ) -> BenchmarkJob:
        status, retries = None, 0
        while status != "success" and retries < max_retries:
            status = self.get_benchmark_job_status(job_id=job_id)["status"]

            if status not in {"success", "running"}:
                raise RuntimeError(
                    f"Benchmark job '{job_id}' returned unexpected status '{status}'."
                )
            logging.info(f"Results for benchmark job '{job_id}' are not yet available.")
            retries += 1
            time.sleep(patience)

        if retries >= max_retries:
            raise RuntimeError(
                f"Maximum number of retries ({max_retries}) exceeded while waiting for results of benchmarking job '{job_id}'."
            )

        return self.get_benchmark_job(job_id=job_id)

    def benchmark(
        self,
        edge_device: str,
        dataset: list[Path] | list[tuple[str, BytesIO]],
        model: Path | tuple[str, BytesIO],
        inference_client_config: Union[TritonInferenceClientConfig],
        model_metadata: Path | tuple[str, BytesIO] | None = None,
        labels: Path | tuple[str, BytesIO] | None = None,
    ) -> tuple[dict[str, list], dict[str, list[Any]]]:
        # 1. Upload benchmark data
        benchmark_data = self.upload_benchmark_data(
            dataset=dataset, model=model, model_metadata=model_metadata, labels=labels
        )

        # 2. Get the bucket name of benchmark data
        benchmark_job_id = benchmark_data.bucket_name

        # 3. Start a benchmark job on that bucket
        self.start_benchmark_job(
            job_id=benchmark_job_id,
            edge_device_config=EdgeDeviceConfig(host=edge_device),
            inference_client_config=inference_client_config,
        )

        # 4. Wait for the benchmark results to become available
        benchmark_job = self.get_benchmark_job_results(job_id=benchmark_job_id)

        # 5. Return result fields
        return benchmark_job.benchmark_results, benchmark_job.inference_results
