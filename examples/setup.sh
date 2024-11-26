#!/usr/bin/env bash

# In Jupyter Notebook terminal, stop running virutal env
deactivate
conda deactivate

set -e

# Download and install Edge Benchmarking Client
git clone https://gitlab.edvsz.hs-osnabrueck.de/agrifood-tef/edge-benchmarking-client.git
cd edge-benchmarking-client
pip3 install -e .
pip3 install torch # to get nn.functional

cd examples
cp .env.example .env

# Add API credentials and run example(s)
echo "Edit .env and add API credentials EDGE_FARM_API_BASIC_AUTH_USERNAME, EDGE_FARM_API_BASIC_AUTH_PASSWORD."
echo ""
echo "After that you can either run main.py or step through the Jupyter Notebook 'main.ipynb'"