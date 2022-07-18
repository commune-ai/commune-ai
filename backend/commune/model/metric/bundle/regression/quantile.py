from functools import partial
from commune.model.metric import (smooth_l1_loss,
                         diff_root_loss,
                         compute_mse,
                          single_quantile_loss)

def v0(cfg,metrics):

    """calculate the metrics"""


    for output_name in cfg['predicted_columns']:

        for time_mode in cfg['metric']['time_modes']:

            metrics[f"S1_{time_mode}_{output_name}"] = \
                dict(fn=nn.SmoothL1Loss(),
                     args=dict(input=f"pred_{time_mode}_{output_name}-mean",
                               target=f"gt_{time_mode}_{output_name}"),
                     w=cfg['loss_weight']['mse'])

            metrics[f"MSE_{time_mode}_{output_name}"] = \
                dict(fn=partial(compute_mse, reduce_dims=[0, 1]),
                     args=dict(y_pred=f"pred_{time_mode}_{output_name}-mean",
                               y_target=f"gt_{time_mode}_{output_name}"))
            for q in cfg['quantiles']:
                metrics[f"Quantile_Loss_{time_mode}_{output_name}_Q-{q}"] = \
                    dict(fn=partial(single_quantile_loss, quantile=q),
                         args=dict(y_pred=f"pred_{time_mode}_{output_name}_Q-{q}",
                                   target=f"gt_{time_mode}_{output_name}"),
                         w=cfg['loss_weight']['quantile'])
