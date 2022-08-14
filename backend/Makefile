
launcher:
	python commune/process/base/launcher.py 
data:
	python3 commune/data/regression/crypto/sushiswap/dataset.py
buckets:
	python commune/client/minio/create_bucket.py
train:
	python commune/experiment/regression/crypto/module.py	
inference: 
	python commune/trainer/regression/crypto/inference.py

market:
	python commune/contract/market/portfolio/task/spawnMultipleTraders.py
demo_commune:
	python commune/contract/commune/base/demo.py
extract_sushi:
	python commune/extract/crypto/sushiswap.py

app_postgres:
	./scripts/app/postgres.sh
app_portfolio:
	./scripts/app/portfolio.sh
app_api:
	streamlit run commune/api/graphql/app.py
sandbox:
	streamlit run sandbox.py 

start: 
	ray start --head 
stop:
	ray stop
restart:
	ray stop; ray start --head

graphql_api: 
	uvicorn commune.api.graphql.main:app --reload --port 8000 --host 0.0.0.0 --app-dir=/app

bittensor:
	streamlit run commune/bittensor/module.py

pip_freeze:
	pip freeze > requirements.txt

purge:
	docker-stop $(docker ps -a -q) && docker rm $(docker ps -a -q);
api:
	python3 commune/api/main.py 

kill_streamlit:
	pkill -9 streamlit