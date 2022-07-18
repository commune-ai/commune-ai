from functools import partial
from commune.model.metric import (smooth_l1_loss,
                         diff_root_loss,
                         compute_mse)








def v0(cfg,metrics):

    """calculate the metrics"""

    for output_name in cfg['predicted_columns']:




        for time_mode in cfg['metric']['time_modes']:

            if 'loss_weight' in cfg['metric']:


                if 'log_prob' in cfg['metric']['loss_weight']:
                    metrics[f"LOG PROB_{time_mode}_{output_name}"]= \
                                                        dict(fn=lambda y, pred_dist: pred_dist.loss(y=y).mean(dim=1),
                                                            args=dict(pred_dist=f"pred_{time_mode}_{output_name}_distribution",
                                                                      y=f"gt_{time_mode}_{output_name}"),
                                                            w= cfg['metric']['loss_weight']['log_prob'])

                if 'root' in cfg['metric']['loss_weight']:
                    metrics[f"ROOT_{time_mode}_{output_name}"]= \
                                                        dict(fn=partial(diff_root_loss,reduce_dims=[1]),
                                                            args=dict(input=f"pred_{time_mode}_{output_name}-mean",
                                                                      target=f"gt_{time_mode}_{output_name}"),
                                                            w= cfg['metric']['loss_weight']['root'])

                if 's1' in cfg['metric']['loss_weight']:
                    metrics[f"S1_{time_mode}_{output_name}"]= \
                                                        dict(fn=partial(smooth_l1_loss, reduce_dims=[1]),
                                                            args=dict(input=f"pred_{time_mode}_{output_name}-mean",
                                                                      target=f"gt_{time_mode}_{output_name}"),
                                                            w= cfg['metric']['loss_weight']['s1'])



            metrics[f"MSE_{time_mode}_{output_name}"]= \
                                                dict(fn=partial(compute_mse, reduce_dims=[1]),
                                                    args=dict(y_pred=f"pred_{time_mode}_{output_name}-mean",
                                                              y_target=f"gt_{time_mode}_{output_name}"))