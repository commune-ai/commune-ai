# Trainer

The trainer class is responsible for creating an appropriate training environment based
on the provided [config](../config/trainer/regression/crypto/base.yaml).

## Using Tags to Organize Experiments


### Experiment Type and Subtype Tags
We want to define experiments based on the following
- **Type**: Type of problem
    - example: classification, regression
- **SubType**: SubType of problem being adressed by the experiment
    - example: crypto, anomoly-detection
  
### Coin and Model Tags
- In our current experiment we also tag what tokens are within the experiment
as well as the model keys. 
- This enables us to create experiments with multiple tokens,
as well as with multiple models, while being able to filter our the models.


