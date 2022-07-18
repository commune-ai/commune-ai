from ray import tune


hyperparams =    {
            # optimizer toggle
            "gp.optimizer.lr": tune.loguniform(1e-3, 1e-1),
            "gp.optimizer.weight_decay": tune.loguniform(1e-4, 1e-1),
            "gp.optimizer.amsgrad": tune.choice([True, False]),

            # model variables
            "gp.embedding_size": tune.choice([4, 8, 16, 32]),
            "gp.num_training_steps": tune.choice([10, 20, 50]),
            "gp.use_ard": tune.choice([True, False]),
        }