
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
app_sandbox:
	streamlit run sandbox.py 

start: 
	ray start --head 
stop:
	ray stop
restart:
	ray stop; ray start --head

graphql_api: 
	uvicorn commune.api.graphql.main:app --reload --port 8000 --host 0.0.0.0 --app-dir=/app


pip_freeze:
	pip freeze > requirements.txt
