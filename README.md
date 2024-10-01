# Edge Benchmarking Client

1. Download the `EdgeBenchmarkingClient` from GitHub:

    ```bash
    git clone https://github.com/agrifoodTEF-de/edge-benchmarking-client.git \
        && cd edge-benchmarking-client
    ```

2. Install the client as a standalone Python module:

    ```bash
    poetry lock; poetry install
    ```
3. Change into the `examples` directory:

    ```bash
    cd examples
    ```

4. Setup the client's runtime environment by editing the `.env.example` file:

    ```bash
    EDGE_FARM_API_PROTOCOL=https
    EDGE_FARM_API_HOST=api.edge-farm.agrifood-tef.edvsz.hs-osnabrueck.de
    EDGE_FARM_API_BASIC_AUTH_USERNAME=admin
    EDGE_FARM_API_BASIC_AUTH_PASSWORD=<api-basic-auth-password>
    ```

    After editing, rename the file to `.env`:

    ```bash
    mv .env.example .env
    ```

5. Place your trained AI model and dataset inside the `examples` directory (like `densenet_onnx`).

6. Within `main.py`, edit `EDGE_DEVICE_HOST` to select an edge device (via its `hostname`) to run the benchmarking job on. You can request all devices managed by the farm manager using the `GET /device/header` endpoint:

    ```bash
    set -a
    source examples/.env
    curl -u "${EDGE_FARM_API_BASIC_AUTH_USERNAME}:${EDGE_FARM_API_BASIC_AUTH_PASSWORD}" \
        "${EDGE_FARM_API_PROTOCOL}://${EDGE_FARM_API_HOST}/device/header"
    ```

    This API request should return a list of known edge devices:

    ```json
    [
        {
            "ip": "192.168.1.3",
            "name": "NVIDIA Jetson AGX Orin 64GB Developer Kit",
            "hostname": "edge-03",
            "heartbeat_interval": 3,
            "timestamp": "2024-10-01T14:06:37.441271",
            "online": true
        },
        ...
    ]
    ```

7. Within `main.py`, edit `EXAMPLE_ROOT_DIR` to point to the name of the parent directory containing your AI model and dataset.

8. Execute `main.py` to assemble and start a benchmarking job using the dataset in `EXAMPLE_ROOT_DIR` on the device `EDGE_DEVICE_HOST`:

    ```bash
    python3 main.py
    ```