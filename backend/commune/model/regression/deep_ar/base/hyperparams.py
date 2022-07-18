from ray import tune


hyperparams ={
    # optimizer toggle
    "optimizer.lr": tune.loguniform(1e-4, 1e-2),
    "optimizer.weight_decay": tune.loguniform(1e-6, 1e-2),
    "optimizer.amsgrad": tune.choice([True, False]),

    # stocastic variables
    "deep_ar.dropout": tune.uniform(0.1, 0.5),
    "deep_ar.rnn_layers": tune.choice([1, 2, 3, 4]),
    "deep_ar.embedding_size": tune.choice([4, 8, 16, 32]),
    "deep_ar.hidden_size": tune.choice([8, 16, 32, 64]),

}