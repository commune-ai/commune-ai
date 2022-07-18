# Models

## Block
- We want to build model blocks so we can compose them into final (**Complete Models**)
- Currently Available Model Blocks
  - gp: gaussian process
  - nbeats: NBEATS based neural networks
  - transformer: transformers with time2vec
  - temporal transfusion: Temporal Transfusion
  - deep ar: deep autoregressive model

## Complete
- Contains final models that can combine different components
- each final model defines the following methods
    - calculate metrics: calculate the metrics from the model
    - define_metrics: define metrics and loss functions for model object
    - learning_step: one step for learning

## Metrics
- Contains custom metric functions for metrics and loss functions

## Distrubution
- Contains Distribution Layers for Distributions as Outputs
