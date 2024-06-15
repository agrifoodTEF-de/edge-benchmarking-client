import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import requests
import validators

from pathlib import Path
from requests.auth import HTTPBasicAuth

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

    def find_dataset(self, root_dir: str, file_extensions: set[str]) -> list[Path]:
        sample_filepaths, root_path = self._collect_files(
            root_dir=root_dir, file_extensions=file_extensions
        )
        logging.info(
            f"Found dataset containing {len(sample_filepaths)} samples with type(s) {file_extensions} in '{root_path}'."
        )

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

    def benchmark(self, dataset: list[Path], model: Path, model_metadata: Path) -> None:
        raise NotImplementedError()
