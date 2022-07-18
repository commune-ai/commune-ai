from ray import tune

hyperparams= {
    # # optimizer toggle
    # "model.oracle.model.transformer.optimizer.lr": tune.loguniform(1e-4, 1e-1),
    # "model.oracle.model.transformer.optimizer.weight_decay": tune.loguniform(1e-5, 1e-2),
    # "model.oracle.model.transformer.optimizer.amsgrad": tune.choice([True, False]),
    # # stocastic variables
    # "model.oracle.model.transformer.transformer.d_model": tune.choice([8, 16, 32, 64]),
    # "model.oracle.model.transformer.transformer.attn_heads": tune.choice([4, 8]),
    # "model.oracle.model.transformer.transformer.dropout": tune.uniform(0.1, 0.8),
    # "model.oracle.model.transformer.transformer.d_ff": tune.choice([8, 16, 32]),
    # "model.oracle.model.transformer.transformer.num_layers": tune.choice([2, 4, 8,  16, 32]),
    # "model.oracle.model.transformer.positional": tune.choice([True, False]),
    # "model.oracle.model.transformer.embedding_size": tune.choice([4, 8, 16, 32]),
    # # "data.pipeline.sample_noise.params.scale": tune.uniform(1, 5),
    # # "data.pipeline.sample_noise.params.shift": tune.uniform(1, 5),
    # # "data.periods.output": tune.choice([6, 12])
    # # optimizer toggle
    # "model.oracle.model.gp.gp.optimizer.lr": tune.loguniform(1e-3, 1e-1),
    # "model.oracle.model.gp.gp.optimizer.weight_decay": tune.loguniform(1e-4, 1e-1),
    # "model.oracle.model.gp.gp.optimizer.amsgrad": tune.choice([True, False]),
    #
    # # model variables
    # "model.oracle.model.gp.gp.embedding_size": tune.choice([4, 8, 16, 32]),
    # "model.oracle.model.gp.gp.num_training_steps": tune.choice([10, 20, 50]),
    # "model.oracle.model.gp.gp.use_ard": tune.choice([True, False]),
    # # data variables
    # "data.batch_size": tune.choice([ 256, 512, 1024]),
    #
    # "data.periods.output": tune.choice([6, 12])
}