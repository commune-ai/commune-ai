from ray import tune

hyperparams ={
  
      # optimizer toggle
      "optimizer.lr": tune.loguniform(1e-3, 1e-1),
      "optimizer.weight_decay": tune.loguniform(1e-4, 1e-2),
      "optimizer.amsgrad": tune.choice([True, False]),

      # stocastic variables
      "nbeats.blocks.trend.dropout": tune.uniform(0.2, 0.4),
      "nbeats.blocks.trend.units": tune.choice([8, 16, 32]),
      "nbeats.blocks.trend.num_block_layers": tune.choice([4, 8, 16]),

      "nbeats.blocks.seasonal.dropout": tune.uniform(0.2, 0.4),
      "nbeats.blocks.seasonal.units": tune.choice([8, 16, 32]),
      "nbeats.blocks.seasonal.num_block_layers": tune.choice([4, 8, 16]),

      "nbeats.embedding_size": tune.choice([4, 8, 16, 32]),
      # "data.batch_size": tune.choice([256, 512, 1024, 2048]),
      # "data.pipeline.post_sample.params.scale": tune.uniform(0, 2),
    }