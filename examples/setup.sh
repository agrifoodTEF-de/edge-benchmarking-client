#!/usr/bin/env bash

set -e

# Download Edge Benchmarking Client and poetry
git clone https://gitlab.edvsz.hs-osnabrueck.de/agrifood-tef/edge-benchmarking-client
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/jovyan/.local/bin:$PATH"

# Install
cd edge-benchmarking-client
poetry lock
poetry install

cd examples
cp .env.example .env

# Add API credentials and run example(s)
echo "Edit .env and add API credentials EDGE_FARM_API_BASIC_AUTH_USERNAME, EDGE_FARM_API_BASIC_AUTH_PASSWORD."
echo ""
echo "After that you can either run main.py or step through the Jupyter Notebook 'main.ipynb'"