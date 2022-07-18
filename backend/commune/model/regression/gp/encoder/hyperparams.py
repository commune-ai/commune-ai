
from ray import tune

hyperparams = {
        # optimizer toggle
        "model.oracle.gp.optimizer.lr": tune.loguniform(1e-3, 1e-1),
        "model.oracle.gp.optimizer.weight_decay": tune.loguniform(1e-4, 1e-1),
        "model.oracle.gp.optimizer.amsgrad": tune.choice([True, False]),

        # model variables
        "model.oracle.gp.embedding_size": tune.choice([4, 8, 16, 32]),
        "model.oracle.gp.num_training_steps": tune.choice([10, 20, 50]),
        "model.oracle.gp.use_ard": tune.choice([True, False]),

        # data variables
        "data.batch_size": tune.choice([64, 128, 256, 512, 1024]),

        # optimizer toggle
        "model.oracle.optimizer.lr": tune.loguniform(1e-4, 1e-2),
        "model.oracle.optimizer.weight_decay": tune.loguniform(1e-6, 1e-2),
        "model.oracle.optimizer.amsgrad": tune.choice([True, False]),

        # encoder parameters
        "model.oracle.encoder.d_model": tune.choice([8, 16, 32, 64]),
        "model.oracle.encoder.attn_heads": tune.choice([4, 8]),
        "model.oracle.encoder.dropout": tune.uniform(0.1, 0.8),
        "model.oracle.encoder.d_ff": tune.choice([8, 16, 32]),
        "model.oracle.encoder.num_layers": tune.choice([2, 4]),
        "model.oracle.encoder.positional": tune.choice([True, False]),
        "model.oracle.encoder.embedding_size": tune.choice([4, 8, 16, 32]),
        "model.oracle.encoder.output_dim": tune.choice([4, 8, 16]),

        # oracle parameters
        "model.oracle.encoder_attn_heads": tune.choice([2, 4]),
        "model.oracle.predict_past": tune.choice([True, False]),

        "data.periods.output": tune.choice([6, 12])
      }