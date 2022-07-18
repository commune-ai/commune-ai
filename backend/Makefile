
launcher:
	python commune/process/base/launcher.py 
data:
	python commune/data/regression/crypto/sushiswap/dataset.py
buckets:
	python commune/client/minio/create_bucket.py
train:
	python commune/experiment/regression/crypto/module.py	
inference: 
	python commune/trainer/regression/crypto/inference.py
traders:
	python commune/contract/model/portfolio/task/spawnMultipleTraders.py
market:
	python commune/contract/market/portfolio/task/spawnMultipleTraders.py
demo_commune:
	python commune/contract/commune/base/demo.py
demo_market:
	./scripts/setup_demo_market.sh
extract_sushi:
	python commune/extract/crypto/sushiswap.py
app_experiment:
	streamlit run commune/app/run_app_experiment.py
app_postgres:
	./scripts/app/postgres.sh
app_portfolio:
	./scripts/app/portfolio.sh
app_api:
	streamlit run commune/api/graphql/app.py

app_sandbox:
	streamlit run commune/app/run_app_sandbox.py 
explain_experiment:
	python commune/explain/complete/regression/crypto/experiment.py
backtest:
	python commune/validate/crypto/backtest.py
debug_graphql:
	python commune/graph_ql/test.py
pipeline: 
	python commune/data/regression/crypto/sushiswap/pipeline.py

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