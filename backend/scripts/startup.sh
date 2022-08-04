#/bin/bash

# pip install -e ./bittensor
# pip install -e ./ipfsspec
python3 commune/client/minio/create_bucket.py
# brownie compile
chmod +x ./scripts/*
# brownie networks add Development dev cmd=ganache-cli host=http://ganache:8545
ray start --head
pip install -e ./bittensor/
alias python=python3
# start 
# uvicorn commune.api.graphql.main:app --reload --port 8001 --host 0.0.0.0 --app-dir=/app

