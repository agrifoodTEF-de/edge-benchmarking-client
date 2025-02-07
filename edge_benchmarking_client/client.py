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
from itertools import islice
from requests import Response
from requests.auth import HTTPBasicAuth
from edge_benchmarking_client.endpoints import (
    DEVICE,
    BENCHMARK_JOB,
    BENCHMARK_DATA,
    BENCHMARK_DATA_MODEL,
    BENCHMARK_DATA_DATASET,
)
from edge_benchmarking_types.edge_farm.models import (
    EdgeDevice,
    BenchmarkModel,
    BenchmarkData,
    InferenceClient,
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
        url = self.api + Path(*paths).as_posix()
        if query is not None:
            query = {k: v for k, v in query.items() if v is not None}
            if query:
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
        chunk_size: int | None = None,
    ) -> BenchmarkData:
        # 1. Create benchmark bucket
        bucket_name: str = self.create_benchmark_bucket()

        # 2. Upload benchmark model data
        benchmark_model: BenchmarkModel = self.upload_benchmark_model(
            bucket_name=bucket_name,
            model=model,
            model_metadata=model_metadata,
            labels=labels,
        )

        # 3. Upload benchmark dataset
        benchmark_dataset: list[str] = self.upload_benchmark_dataset(
            bucket_name=bucket_name,
            dataset=dataset,
            chunk_size=chunk_size,
        )

        benchmark_data = BenchmarkData(
            bucket_name=bucket_name, model=benchmark_model, dataset=benchmark_dataset
        )
        return benchmark_data

    def create_benchmark_bucket(self, bucket_name: str | None = None) -> str:
        response = requests.post(
            url=self._endpoint(BENCHMARK_DATA, query={"bucket_name": bucket_name}),
            auth=self.auth,
        )
        response.raise_for_status()
        bucket_name = response.json()
        logging.info(f"{response.status_code} - {bucket_name}")
        return bucket_name

    def upload_benchmark_model(
        self,
        bucket_name: str,
        model: Path | tuple[str, BytesIO],
        model_metadata: Path | tuple[str, BytesIO] | None = None,
        labels: Path | tuple[str, BytesIO] | None = None,
    ) -> BenchmarkModel:
        benchmark_model_files = []
        try:
            model_data = (
                ("model", model)
                if isinstance(model, tuple)
                else ("model", (model.name, open(model, "rb")))
            )
            benchmark_model_files.append(model_data)

            if model_metadata is not None:
                model_metadata = (
                    ("model_metadata", model_metadata)
                    if isinstance(model_metadata, tuple)
                    else (
                        "model_metadata",
                        (model_metadata.name, open(model_metadata, "rb")),
                    )
                )
                benchmark_model_files.append(model_metadata)

            if labels is not None:
                labels_data = (
                    ("labels", labels)
                    if isinstance(labels, tuple)
                    else ("labels", (labels.name, open(labels, "rb")))
                )
                benchmark_model_files.append(labels_data)

            for field_name, (filename, payload) in benchmark_model_files:
                payload.seek(os.SEEK_END)
                if not payload.tell():
                    raise ValueError(
                        f"Benchmark model file '{(field_name, filename)}' is empty."
                    )
                payload.seek(os.SEEK_SET)

            response = requests.patch(
                url=self._endpoint(
                    BENCHMARK_DATA_MODEL, query={"bucket_name": bucket_name}
                ),
                files=benchmark_model_files,
                auth=self.auth,
            )
            response.raise_for_status()
            benchmark_model = BenchmarkModel.model_validate(response.json())
            logging.info(f"{response.status_code} - {benchmark_model}")
            return benchmark_model
        finally:
            for _, (_, payload) in benchmark_model_files:
                if isinstance(payload, BytesIO):
                    payload.seek(os.SEEK_SET)
                else:
                    payload.close()

    def upload_benchmark_dataset(
        self,
        bucket_name: str,
        dataset: list[Path] | list[tuple[str, BytesIO]],
        chunk_size: int | None = None,
    ) -> list[str]:
        assert len(dataset), "Dataset is empty."
        if chunk_size is not None and chunk_size > 1:
            dataset = [
                dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)
            ]
            logging.info(
                f"Uploading dataset in {len(dataset)} chunks of size {chunk_size}."
            )
        else:
            dataset = [dataset]

        benchmark_dataset = []
        for chunk in dataset:
            dataset_data = []
            try:
                for sample in chunk:
                    if not isinstance(sample, tuple):
                        sample = (sample.name, open(sample, "rb"))
                    sample = ("dataset", (sample))
                    dataset_data.append(sample)

                for field_name, (filename, payload) in dataset_data:
                    payload.seek(os.SEEK_END)
                    if not payload.tell():
                        raise ValueError(
                            f"Benchmark dataset sample file '{(field_name, filename)}' is empty."
                        )
                    payload.seek(os.SEEK_SET)

                response = requests.patch(
                    url=self._endpoint(
                        BENCHMARK_DATA_DATASET, query={"bucket_name": bucket_name}
                    ),
                    files=dataset_data,
                    auth=self.auth,
                )
                response.raise_for_status()
                benchmark_dataset_chunk = response.json()
                logging.info(f"{response.status_code} - {benchmark_dataset_chunk}")
                benchmark_dataset += benchmark_dataset_chunk
            finally:
                for _, (_, payload) in dataset_data:
                    if isinstance(payload, BytesIO):
                        payload.seek(os.SEEK_SET)
                    else:
                        payload.close()
        return benchmark_dataset

    def start_benchmark_job(
        self,
        job_id: str,
        edge_device: EdgeDevice,
        inference_client: InferenceClient,
    ) -> Response:
        response = requests.post(
            url=self._endpoint(BENCHMARK_JOB, job_id, "start"),
            json={
                "edge_device": edge_device.model_dump(),
                "inference_client": inference_client.model_dump(),
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

    def remove_benchmark_job(self, job_id: str) -> Response:
        response = requests.delete(
            url=self._endpoint(BENCHMARK_JOB, job_id), auth=self.auth
        )
        response.raise_for_status()
        logging.info(f"{response.status_code}")
        return response

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
        inference_client: InferenceClient,
        model_metadata: Path | tuple[str, BytesIO] | None = None,
        labels: Path | tuple[str, BytesIO] | None = None,
        chunk_size: int | None = None,
        cleanup: bool = True,
    ) -> BenchmarkJob:
        benchmark_job_id = None
        try:
            # 1. Upload benchmark data
            benchmark_data = self.upload_benchmark_data(
                dataset=dataset,
                model=model,
                model_metadata=model_metadata,
                labels=labels,
                chunk_size=chunk_size,
            )

            # 2. Get the bucket name of benchmark data
            benchmark_job_id = benchmark_data.bucket_name

            # 3. Start a benchmark job on that bucket
            self.start_benchmark_job(
                job_id=benchmark_job_id,
                edge_device=EdgeDevice(host=edge_device),
                inference_client=inference_client,
            )

            # 4. Wait for the benchmark results to become available
            benchmark_job = self.get_benchmark_job_results(job_id=benchmark_job_id)

            # 5. Return benchmark job containing job id and results
            return benchmark_job
        finally:
            if benchmark_job_id is not None and cleanup:
                # Ensure that all benchmark job artifacts are cleaned up
                self.remove_benchmark_job(job_id=benchmark_job_id)
