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
from email.parser import HeaderParser
from requests.auth import HTTPBasicAuth

from edge_benchmarking_client.endpoints import (
    DEVICE,
    BENCHMARK_JOB,
    BENCHMARK_DATA,
    BENCHMARK_DATA_MODEL,
    BENCHMARK_DATA_DATASET,
    BENCHMARK_DATA_DATASET_CAPTURE,
)
from edge_benchmarking_types.edge_farm.models import (
    EdgeDevice,
    BenchmarkModel,
    BenchmarkData,
    InferenceClient,
    ExternalDataProvider,
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
        root_dir: str, file_extensions: set[str] | None
    ) -> tuple[list[Path], Path]:
        root_path = Path(root_dir).expanduser().resolve()
        patterns = {f"*{ext}" for ext in file_extensions} if file_extensions else {"*"}
        filepaths = sorted(
            {p for pattern in patterns for p in root_path.rglob(pattern)}
        )
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

    @staticmethod
    def _file_is_bytes(file: Path | tuple[str, BytesIO]) -> bool:
        return (
            isinstance(file, tuple)
            and len(file) == 2
            and isinstance(file[0], str)
            and isinstance(file[1], BytesIO)
        )

    def _create_benchmark_bucket(self, bucket_name: str | None = None) -> str:
        response = requests.post(
            url=self._endpoint(BENCHMARK_DATA, query={"bucket_name": bucket_name}),
            auth=self.auth,
        )
        response.raise_for_status()
        bucket_name = response.json()
        logging.info(f"{response.status_code} - {bucket_name}")
        return bucket_name

    def _upload_benchmark_model(
        self,
        bucket_name: str,
        model: Path | tuple[str, BytesIO],
        model_metadata: Path | tuple[str, BytesIO] | None = None,
        labels: Path | tuple[str, BytesIO] | None = None,
    ) -> BenchmarkModel:
        fields = {"model": model}

        if model_metadata is not None:
            fields["model_metadata"] = model_metadata

        if labels is not None:
            fields["labels"] = labels

        response = self._upload_benchmark_files(
            endpoint=BENCHMARK_DATA_MODEL, fields=fields, bucket_name=bucket_name
        )
        benchmark_model = BenchmarkModel.model_validate(response.json())
        logging.info(f"{response.status_code} - {benchmark_model}")
        return benchmark_model

    def _upload_benchmark_files(
        self,
        endpoint: str,
        fields: (
            dict[str, Path]
            | dict[str, list[Path]]
            | dict[str, list[tuple[str, BytesIO]]]
        ),
        bucket_name: str,
    ) -> list[str]:
        try:
            upload_files = []
            for field_name, files in fields.items():
                if not isinstance(files, list):
                    files = [files]
                upload_files += [
                    (
                        field_name,
                        (
                            file
                            if self._file_is_bytes(file)
                            else (file.name, open(file, "rb"))
                        ),
                    )
                    for file in files
                ]

            for name, (filename, payload) in upload_files:
                payload.seek(os.SEEK_END)
                if not payload.tell():
                    raise ValueError(f"File '{(name, filename)}' is empty.")
                payload.seek(os.SEEK_SET)

            response = requests.patch(
                url=self._endpoint(endpoint, query={"bucket_name": bucket_name}),
                files=upload_files,
                auth=self.auth,
            )
            response.raise_for_status()
            return response
        finally:
            for _, (_, payload) in upload_files:
                if isinstance(payload, BytesIO):
                    payload.seek(os.SEEK_SET)
                else:
                    payload.close()

    def _upload_benchmark_dataset(
        self,
        bucket_name: str,
        dataset: list[Path] | list[tuple[str, BytesIO]],
        chunk_size: int | None = None,
    ) -> list[str]:
        def _upload_benchmark_dataset_files() -> list[str]:
            if chunk_size is not None and chunk_size > 1 and len(dataset) > chunk_size:
                dataset_chunks = [
                    dataset[i : i + chunk_size]
                    for i in range(0, len(dataset), chunk_size)
                ]
                logging.info(
                    f"Uploading dataset in {len(dataset_chunks)} chunks of size {chunk_size}."
                )
            else:
                dataset_chunks = [dataset]

            filepaths: list[str] = []
            for dataset_chunk in dataset_chunks:
                response = self._upload_benchmark_files(
                    endpoint=BENCHMARK_DATA_DATASET,
                    fields={"dataset": dataset_chunk},
                    bucket_name=bucket_name,
                )
                filepaths += response.json()
            return filepaths

        def _upload_benchmark_dataset_bytes() -> list[str]:
            return self._upload_benchmark_files(
                endpoint=BENCHMARK_DATA_DATASET,
                fields={"dataset": dataset},
                bucket_name=bucket_name,
            ).json()

        if isinstance(dataset, list):
            assert len(dataset), "List of dataset files is empty."
            if isinstance(dataset[0], Path):
                filepaths = _upload_benchmark_dataset_files()
            elif self._file_is_bytes(dataset[0]):
                filepaths = _upload_benchmark_dataset_bytes()
            else:
                raise TypeError("Unsupported list of dataset samples.")
        else:
            raise TypeError("Unsupported dataset type.")

        return filepaths

    def find_dataset(
        self, root_dir: str, file_extensions: set[str] | None = None
    ) -> list[Path]:
        sample_filepaths, root_path = self._collect_files(
            root_dir=root_dir, file_extensions=file_extensions
        )
        logging.info(
            f"Found dataset containing {len(sample_filepaths)} samples"
            f"{f' with type(s) {file_extensions}' if file_extensions else ''} in '{root_path}'."
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

    def capture_benchmark_data(
        self,
        root_dir: Path,
        external_data_provider: ExternalDataProvider,
        chunk_size: int = 8192,
    ) -> Path:
        with requests.post(
            url=self._endpoint(BENCHMARK_DATA_DATASET_CAPTURE),
            json=external_data_provider.model_dump(),
            auth=self.auth,
            stream=True,
        ) as response:
            response.raise_for_status()
            content_disposition = response.headers.get("Content-Disposition")

            def get_filename(content_disposition: str | None) -> str:
                default_filename = "dataset.zip"
                if not content_disposition:
                    return default_filename
                parser = HeaderParser()
                headers = parser.parsestr(
                    f"Content-Disposition: {content_disposition}\n"
                )
                return headers.get_param(
                    "filename", header="Content-Disposition", failobj=default_filename
                )

            zip_filename = get_filename(content_disposition)
            zip_filepath = root_dir.joinpath(zip_filename)
            root_dir.mkdir(parents=True, exist_ok=True)
            with open(zip_filepath, "wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
        return zip_filepath

    def get_welcome_message(self) -> Response:
        response = requests.get(self.api, auth=self.auth)
        response.raise_for_status()
        logging.info(f"{response.status_code} - {response.json()}")
        return response

    def start_benchmark_job(
        self,
        job_id: str,
        edge_device: EdgeDevice,
        inference_client: InferenceClient,
        cpu_only: bool = False,
    ) -> Response:
        response = requests.post(
            url=self._endpoint(BENCHMARK_JOB, job_id, "start"),
            json={
                "edge_device": edge_device.model_dump(),
                "inference_client": inference_client.model_dump(),
                "cpu_only": cpu_only,
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

    def upload_benchmark_data(
        self,
        dataset: list[Path] | list[tuple[str, BytesIO]],
        model: Path | tuple[str, BytesIO],
        model_metadata: Path | tuple[str, BytesIO] | None = None,
        labels: Path | tuple[str, BytesIO] | None = None,
        chunk_size: int | None = None,
    ) -> BenchmarkData:
        # 1. Create benchmark bucket
        bucket_name: str = self._create_benchmark_bucket()

        # 2. Upload benchmark model data
        benchmark_model: BenchmarkModel = self._upload_benchmark_model(
            bucket_name=bucket_name,
            model=model,
            model_metadata=model_metadata,
            labels=labels,
        )

        # 3. Upload benchmark dataset
        benchmark_dataset: list[str] = self._upload_benchmark_dataset(
            bucket_name=bucket_name,
            dataset=dataset,
            chunk_size=chunk_size,
        )

        benchmark_data = BenchmarkData(
            bucket_name=bucket_name, model=benchmark_model, dataset=benchmark_dataset
        )
        return benchmark_data

    def benchmark(
        self,
        edge_device: str,
        dataset: list[Path] | list[tuple[str, BytesIO]],
        model: Path | tuple[str, BytesIO],
        inference_client: InferenceClient,
        model_metadata: Path | tuple[str, BytesIO] | None = None,
        labels: Path | tuple[str, BytesIO] | None = None,
        chunk_size: int | None = None,
        cpu_only: bool = False,
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
                cpu_only=cpu_only,
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
