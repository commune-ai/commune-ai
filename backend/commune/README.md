
# Components

## Model

- Contains modular model blocks that are composed to build bigger models
- Each model has a **model key** which ponts to its relative path in the models folder
- We can add multiple different models within this folder, and call them via the model key

## Dataset

There are two components to this Module
- **Data Manager**
    - Calls binance api for any token
    - Saves/Loads already extracted and processed data
        - this avoids the need to reload previously retrieved results
    - Loads data from data manager which loads a dataframe with 1 row per hour of ticker data

- **Data Loader**
    - Training and Testing Set is Split by a specified Date in the **config**
    - Additional Preprocessing on column level specified in **config.data.transform**

      
## Transformations
This contains transformations that are standardized with the following format
- **fit**: fit the transformation parameters (save the parameters)
- **transform**: transform the column

- ### [column/split level](transformation/preprocessing_store.py): 
    - apply to the entire split 
    - examples: standard deviation, temporal difference, log tranform etc7

- ### [sample level](transformation/sample_processing_store.py)
    - Each sample has a specific input and output period as defined in
      the **configs.data.periods**

## Trainer
The trainer has the following roles 
- handling an environment for training as well as inference
- Connecting MLFLOW for storing
    - experiment tags
    - run tags
    - metrics
    - saved outputs
        - this typically includes a dictionary of batched tensors
    - model checkpoints
    
- connects with dataloader and models bia the specified config

## App
- This folder contains several streamlit apps
    - [run_app_experiment.py](./run_app_user.py) (Stable)
        - view results across runs and experiments
        - You can also see the predictions for the model
-   - [run_app_user.py](./run_app_user.py) (Stable)
        - shows the input and predicted price using graphql 
        - this demonstrates using the graphql api
    - [run_app_preprocess.py](./run_app_preprocess.py) (WIP)
        - app for visualizing sample and split level transformations preproces
    
### Running App Locally/In Docker Container
- within src/ run ```streamlit run run_app_*.py```
- open the port via localhost:{port of streamlit app}
- if you are running the stramlit app via a docker container, make sure to expose the port
if you are running the app locally make sure the mlruns folder is in the same (backend/src)
## Graph QL
- contains models (schema) and query when pulling in tokens and model predictions

# Running and Experiment

- to run an experiment, you will need to call **run_experiments.py** while specifying the following arguments:
    - **exp_name**: 
        - Name of the Experiment
        - If the experiment already exists, it adds runs to that experiment
    - **config**: 
      - we specify the config ([found here](config/block))
    - **trials**: 
        - number of trials
        - if > 1, then hyperoptimization is used via ray tune
    - **token_pairs**: 
      - list of at least one token pair
      - If more token pairs are specified, the model will train on multiple tokens 
        - at the moment training on multiple tokens does not appear to increase performance
    
```bash
python run_experiments.py --config ORACLE_GP_NBEATS_TRANSFORMER --trials 1 --exp_name DEMO --token_pairs BTCUSDT

```

    
