# Welcome to the Model Backend


## Experiment Layout Summary

We want to think of logging our experiments into 3 seperate levels


     
###Experiment Info

1. **Type**: the general problem  we are trying to solve (ie. regression)
2. **SubType**: a subtype of the general type (ie. price_prediction)
3. **Experiment**: A collection of runs attempting to solve a task
4. **Run**: A single model trained to learn a task
   - Each run can represent a different set of hyperparameters
    
```angular2html

Type: regression
    SubType: crypto
        Experiment
            Run

```

### Run Info     
Each run will contain information we will use later when selecting a model
- **Model State**: Saved variables of model(s) involved in the run
- **Samples**: Saved samples from training and evaluation set(s)
- **Config**: Configuration of the training/inference environment
- **Metrics**: Metrics used to evaluate model performance
- **Parameters**: Parameters


## Setting Up Environment


###Docker Compose

file: [docker-compose.yml](docker-compose.yml)
envs: [.env](.env)
- Starts the following services based on 
    - **mongo**
        - **Databases**
            - **features**: Stored features during extraction/preprocessing steps
            - **model**: Stored model embeddings and predictions
    - **postgres**
      - **Databases**:
          - mlflow: used to store mlflow experiment hyperparameters, tags, and metrics
    - **s3/minio**
        - mlflow: stores experiments objects from each run
        - backend: stores artifacts from pipeline into backend
    - **mlflow server**
        - uses postgres/mysql for params/tags/metrics 
        - s3/minio for artifacts
    - **backend**
        - python environment for experiments and graphQL/App endpoints
    
To spin up the instance
```angular2html
# spin up the environment
docker-compose up

# spin down the environment
docker-compose down
```

If you have an nvidia gpu, use x

```angular2html
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [ gpu ]
```

   
## Fetching Crypto Data (Required for Experiments)

- The following commands fetches and preprocesses crypto data

- Docker-Compose
    ```angular2html
    docker exec -d backend python cron/extract/crypto.py 
    ```
- Virtual Env 
    ```angular2html
    cd commune-ai/app/backend/src
    python cron/extract/crypto.py
    ```  
    


## Mlflow and Experiments


### Running an Experiment

- to run an experiment, you will need to call **run_experiments.py** while specifying the following arguments:
    - **exp_name**: 
        - Name of the Experiment
        - If the experiment already exists, it adds runs to that experiment
    - **config**: 
      - we specify the config ([found here](./config/templates))
    - **trials**: 
        - number of trials
        - if > 1, then hyperoptimization is used via ray tune
    - **token_pairs**: 
      - list of at least one token pair
      - If more token pairs are specified, the model will train on multiple tokens 
        - at the moment training on multiple tokens does not appear to increase performance
   
**Example**
- running in virtual env environment 

    
```bash
docker exec -d backend \
        python run_experiments.py --config oracle_deep_ar \
                                  --trials 100 \
                                  --exp_name DEMO \
                                  --token_pairs ETHUSDT
 ```

- At the moment, experiments are stored locally in [backend/src/mlruns](backend/src/mlruns)
- When you pull a new branch, the mlruns folder will be empty


# Endpoints

To Setup the Endpoings you can run a script  as so:

```bash

docker exec -it backend \
bash -c "chmod +x ./run_endpoints.sh && ./run_endpoints.sh"

```


##  GraphQL Server for Coin Data and Predictions
Setup GraphQL API for querying token data and predictions from experiments
For GraphQl Schema and Query Examples, Go to [src/graph_ql/README.md](src/graph_ql/README.md)


- running in virtual env environment (commune_env)


```bash
docker exec -d backend uvicorn graph_ql.main:app --reload --port 8000 --host 0.0.0.0
```

Go to the API via localhost:8000 and play around with the queries


- Please ensure you have at least one experiment run as well as the loaded crypto data


## Visual Applications Interfaces Via Streamlit
- These apps help us understand our experiments, the preprocessing and what the user
sees in the front end. Feel free to add more application interface
  
### User Interact Application
- [run_app_user.py](src/run_app_user.py) 
- shows the input and predicted price using graphql 
    - this demonstrates using the graphql api
```bash
docker exec -it -d backend streamlit run run_app_user.py --server.port 8501
```

### Experimental Review Interface 
- [run_app_experiment.py](src/run_app_user.py) 
    - view results across runs and experiments
    - You can also see the predictions for the model
```bash
docker exec -it -d backend streamlit run run_app_experiment.py --server.port 8502
```
### Preprocessing Review Application

- [run_app_preprocess.py](src/run_app_preprocess.py) 
    - app for visualizing transformations on Series Data

```bash
docker exec -it -d backend streamlit run run_app_preprocess.py --server.port 8503
```

### Run All the Endpoints

```bash
docker exec -it -d endpoints streamlit run run_app_user.py --server.port 8501 &&
docker exec -it -d endpoints streamlit run run_app_experiment.py --server.port 8502 &&
docker exec -it -d endpoints streamlit run run_app_preprocess.py --server.port 8503 &&
docker exec -d endpoints uvicorn graph_ql.main:app --reload --port 8000 --host 0.0.0.0


```

# Connecting to Remote via ssh

If you dont want to 

```bash
ssh -NfL {REMOTE_PORT}:localhost:{LOCAL_PORT} {USER}:{IP_ADDRESS} 
```

**NOTE**: In order to connect with the endpoints, you need to obtain the user, ipaddress and login credentials. (Contact Moi)






