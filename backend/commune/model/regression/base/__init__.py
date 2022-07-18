import ray
import torch
from commune.model.base import GenericBaseModel
from commune.transformation.block.hash import String2IntHash
from commune.utils.ml import tensor_dict_check, tensor_info_dict
from commune.utils.misc import dict_put
class RegressionBaseModel(GenericBaseModel):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)


    def learning_step(self, x_batch):
        '''
        Calculate the learning step
        '''
        tensor_dict = self.predict(**x_batch)
        tensor_dict.update(x_batch)

        # calculate sample wise matrics
        sample_metric_dict = self.calculate_metrics(tensor_dict)
        self.optimizer.zero_grad()


        sample_metric_dict["total_loss"].mean().backward(retain_graph=True)

        self.optimizer.step()


        sample_metric_dict = ray.get(self.data.stratify_timescale_token_batch.remote(sample_metric_dict,
                                                    token_hash_tensor=tensor_dict['token_hash'],
                                                    timescale_hash_tensor=tensor_dict['timescale_hash']))
        return tensor_dict,sample_metric_dict


    def evaluate(self, x_batch, calculate_metrics=False):
        with torch.no_grad():
            tensor_dict =self.predict(**x_batch)
            tensor_dict.update(x_batch)

            if calculate_metrics:
                sample_score_dict = self.calculate_metrics(tensor_dict)
                sample_score_dict = ray.get(self.data.stratify_timescale_token_batch.remote(sample_score_dict,
                                                                        token_hash_tensor=tensor_dict['token_hash'],
                                                                        timescale_hash_tensor=tensor_dict['timescale_hash']))

                tensor_dict = ray.get(self.data.stratify_timescale_token_batch.remote(tensor_dict,
                                                    token_hash_tensor=tensor_dict['token_hash'],
                                                    timescale_hash_tensor=tensor_dict['timescale_hash']))


                return tensor_dict, sample_score_dict
            else:

                tensor_dict = ray.get(self.data.stratify_timescale_token_batch.remote(tensor_dict,
                                                    token_hash_tensor=tensor_dict['token_hash'],
                                                    timescale_hash_tensor=tensor_dict['timescale_hash']))


                out_dict = {}
                for timescale in tensor_dict.keys():
                    for token, input_dict in tensor_dict[timescale].items():
                        input_dict =  self.get_outputs(input_dict=input_dict, pipeline_map=self.pipeline_map[token])
                        dict_put(out_dict, [timescale, token], input_dict)
                        # print(f'CHECK ML ({token.upper()})',tensor_dict_check({k:v for k,v in input_dict.items() if v.dtype == torch.float32}))

                return out_dict


    def get_outputs(self, input_dict, pipeline_map):


        out_dict = {}
        for time_mode in ["future", 'past']:
            for target in self.targets:
                inv_transform = pipeline_map[target].inverse
                out_keys = [f"pred_{time_mode}_{target}-lower",
                             f"pred_{time_mode}_{target}-mean",
                             f"pred_{time_mode}_{target}-upper",
                             f"gt_{time_mode}_{target}",
                             f"gt_{time_mode}_{target}-raw" ]
                
                

                out_dict.update({k:input_dict[k] for k in out_keys })
                check_out_dict  = {k:out_dict[k] for k in out_keys[-1:]}
                # print(f"ML CHECK (BEFORE) : ",tensor_dict_check(check_out_dict))
                for out_key in out_keys:
                    if out_key in input_dict :
                        if out_key.split('-')[-1] == 'raw':
                            out_dict[out_key] = input_dict[out_key]
                        else:
                            out_dict[out_key] = inv_transform(input_dict[out_key])
                    
                # print(f"ML CHECK (AFTER) : ",tensor_dict_check(check_out_dict))


        out_dict['timestamp'] = input_dict['timestamp']


        return out_dict

    def predict(self, **kwargs):
        return self(**kwargs)

    def connect_data(self):

        cfg = self.cfg

        data = self.data

        model_name = self.model_name

        data_info = ray.get(data.get_info.remote())
        cfg['periods'] = data_info['periods']
        cfg[model_name]['temporal_features'] = data_info['input_columns']
        cfg[model_name]['known_future_features'] = data_info['known_future_features']

        # get categorical featuresx
        categorical_feature_info = data_info['categorical_feature_info']
        categorical_features = list(categorical_feature_info['unique_values'].keys())
        cfg[model_name]['categorical_features'] = categorical_features
        default_embedding_size = cfg[model_name]['embedding_size']
        embedding_sizes = {category: (cardinality, default_embedding_size)
                           for category,cardinality in categorical_feature_info['unique_values_count'].items()}
        
        print(embedding_sizes, "EMBEDDING_SIZES")
        embedding_sizes.update(cfg[model_name]['embedding_sizes'])
        cfg[model_name]['embedding_sizes'] = embedding_sizes

        self.cfg = cfg
        self.data = data
        self.pipeline_map = ray.get(data.get_pipeline_map.remote())
