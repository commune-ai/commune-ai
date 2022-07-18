from ray import tune

hyperparams =    {
      # optimizer toggle
      "optimizer.lr": tune.loguniform(1e-3, 1e-1),
      "optimizer.weight_decay": tune.loguniform(1e-5, 1e-2),
      "optimizer.amsgrad": tune.choice([True, False]),
      # stocastic variables
      "transformer.d_model": tune.choice([8, 16, 32, 64]),
      "transformer.attn_heads": tune.choice([4, 8]),
      "transformer.dropout": tune.uniform(0.1, 0.8),
      "transformer.d_ff": tune.choice([8, 16, 32]),
      "transformer.num_layers": tune.choice([2, 4]),
      "transformer.positional": tune.choice([True, False]),
      "transformer.embedding_size": tune.choice([4, 8, 16, 32]),
      #"data.batch_size": tune.choice([ 256, 512]),
      # "data.pipeline.sample_noise.params.scale": tune.uniform(1, 5),
      # "data.pipeline.sample_noise.params.shift": tune.uniform(1, 5),
      # "data.periods.output": tune.choice([6, 12])

    }