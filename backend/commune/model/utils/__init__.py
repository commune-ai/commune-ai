def scale_model_dropout(model, scale_factor):
    model_train_state = model.training

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            if not model_train_state:
                m.train()
            m.p = scale_factor * m.p

    return model



