from ray import tune


hyperparams ={
      # optimizer toggle
      "model.oracle.optimizer.lr": tune.loguniform(1e-4, 1e-2),
      "model.oracle.optimizer.weight_decay": tune.loguniform(1e-6, 1e-2),
      "model.oracle.optimizer.amsgrad": tune.choice([True, False]),

      # stocastic variables
      "model.oracle.nbeats.blocks.trend.dropout": tune.uniform(0.1, 0.8),
      "model.oracle.nbeats.blocks.trend.units": tune.choice([4, 8, 16, 32]),
      "model.oracle.nbeats.blocks.trend.num_block_layers": tune.choice([2, 4]),

      "model.oracle.nbeats.blocks.seasonal.dropout": tune.uniform(0.1, 0.8),
      "model.oracle.nbeats.blocks.seasonal.units": tune.choice([4, 8, 16, 32]),
      "model.oracle.nbeats.blocks.seasonal.num_block_layers": tune.choice([2, 4]),

      "model.oracle.nbeats.embedding_size": tune.choice([4, 8, 16, 32]),

      "data.batch_size": tune.choice([64, 128, 256]),
      "data.pipeline.sample_noise.params.scale": tune.uniform(1, 5),
      "data.pipeline.sample_noise.params.shift": tune.uniform(1, 5),
      "data.periods.output": tune.choice([6, 12])

    }