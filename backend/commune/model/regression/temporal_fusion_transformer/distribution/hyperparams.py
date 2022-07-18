from ray import tune


hyperparams ={
      # optimizer toggle
      "model.oracle.optimizer.lr": tune.loguniform(1e-4, 1e-2),
      "model.oracle.optimizer.weight_decay": tune.loguniform(1e-6, 1e-2),
      "model.oracle.optimizer.amsgrad": tune.choice([True, False]),

      # model variables
      "model.oracle.temporal_fusion_transformer.hidden_size": tune.choice([8, 16, 32, 64]),
      "model.oracle.temporal_fusion_transformer.attention_head_size": tune.choice([2, 4, 8]),
      "model.oracle.temporal_fusion_transformer.dropout": tune.uniform(0.1, 0.8),
      "model.oracle.temporal_fusion_transformer.lstm_layers": tune.choice([1, 2, 4]),
      "model.oracle.temporal_fusion_transformer.embedding_size": tune.choice([4, 8, 16, 32]),
      "model.oracle.temporal_fusion_transformer.hidden_continuous_size": tune.choice([4, 8, 16, 32]),

      # data variables
      "data.batch_size": tune.choice([64, 128, 256, 512, 1024]),
      # "data.pipeline.sample_noise.params.scale": tune.uniform(1, 5),
      # "data.pipeline.sample_noise.params.shift": tune.uniform(1, 5),
      "data.periods.output": tune.choice([6, 12])

    }