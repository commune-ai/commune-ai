from ray import tune


hyperparams ={
      # optimizer toggle
      "optimizer.lr": tune.loguniform(1e-4, 1e-2),
      "optimizer.weight_decay": tune.loguniform(1e-6, 1e-2),
      "optimizer.amsgrad": tune.choice([True, False]),

      # model variables
      "temporal_fusion_transformer.hidden_size": tune.choice([8, 16, 32, 64]),
      "temporal_fusion_transformer.attention_head_size": tune.choice([2, 4, 8]),
      "temporal_fusion_transformer.dropout": tune.uniform(0.1, 0.8),
      "temporal_fusion_transformer.lstm_layers": tune.choice([1, 2, 4]),
      "temporal_fusion_transformer.embedding_size": tune.choice([4, 8, 16, 32]),
      "temporal_fusion_transformer.hidden_continuous_size": tune.choice([4, 8, 16, 32]),

    }