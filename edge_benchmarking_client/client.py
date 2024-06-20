import io
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import requests
import validators
import pandas as pd

from pathlib import Path
from requests import Response
from requests.auth import HTTPBasicAuth
from edge_benchmarking_client.endpoints import BENCHMARK_DATA, BENCHMARK_JOB_START

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
        response = requests.get(self.api, auth=self.auth)
        response.raise_for_status()
        logging.info(
            f"Edge Farm API is reachable with status code {response.status_code}."
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

    def _endpoint(self, endpoint: str, path: str = "") -> str:
        url = Path(self.api, endpoint, path)
        if not validators.url(url):
            raise RuntimeError(f"Invalid URL: {url}")
        return url

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

    def _upload_benchmark_data(
        self, dataset: list[Path], model: Path, model_metadata: Path
    ) -> Response:
        try:
            benchmark_data_files = [
                ("dataset", (sample.name, open(sample, "rb"))) for sample in dataset
            ] + [
                ("model", (model.name, open(model, "rb"))),
                ("model_metadata", (model_metadata.name, open(model_metadata, "rb"))),
            ]
            response = requests.post(
                url=self._endpoint(BENCHMARK_DATA),
                files=benchmark_data_files,
                auth=self.auth,
            )
            response.raise_for_status()
            logging.info(f"{response.status_code} - {response.json()}")
            return response
        finally:
            for _, (_, fh) in benchmark_data_files:
                fh.close()

    def _start_benchmark_job(self, job_id: str) -> Response:
        response = requests.post(url=self._endpoint(BENCHMARK_JOB_START, job_id))
        response.raise_for_status()
        logging.info(f"{response.status_code} - {response.json()}")
        return response

    def _wait_for_benchmark_results(self, benchmark_job_id: str) -> Response:
        raise NotImplementedError()

    def benchmark(
        self, dataset: list[Path], model: Path, model_metadata: Path
    ) -> pd.DataFrame:
        # 1. Upload benchmarking data
        upload_benchmark_data_response = self._upload_benchmark_data(
            dataset=dataset, model=model, model_metadata=model_metadata
        )

        # 2. Get the bucket name of the benchmarking data
        benchmark_job_id = upload_benchmark_data_response.json()["bucket_name"]

        # 3. Start a benchmarking job on that bucket
        start_benchmark_job_response = self._start_benchmark_job(
            job_id=benchmark_job_id
        )

        # 4. Wait (async?) for the benchmarking results to be available
        benchmark_results_response = self._wait_for_benchmark_results(
            benchmark_job_id=benchmark_job_id
        )

        # 5. Process the benchmarking results (visualize, analyze, ...)
        benchmark_results_csv = io.BytesIO(benchmark_results_response.content)
        benchmark_results_df = pd.read_csv(benchmark_results_csv)

        return benchmark_results_df
