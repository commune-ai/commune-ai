import os
import sys
os.environ['WDIR'] = os.getenv('MODEL_WDIR')
os.environ['NUMEXPR_MAX_THREADS'] = '24'
sys.path.append(os.getenv('PWD'))

import datetime
import numpy as np
import torch
import mlflow
from copy import deepcopy
from deepdiff import DeepDiff
import ray
from commune.data.regression.crypto.sushiswap.dataset import Dataset
from commune.utils.misc import get_object, load_pickle, string_replace, roundTime, chunk, dict_put, dict_has, dict_get, round_sig
from commune.ray.utils import create_actor
from commune.config.utils import resolve_devices
from commune.config import ConfigLoader
from commune.experiment.manager.module import ExperimentManager
from commune.utils import format_token_symbol
from commune.process import BaseProcess
from commune.utils.ml import tensor_dict_check

from commune.ray.utils import create_actor


class InferenceManager(BaseProcess):
    default_cfg_path = f"{os.getenv('PWD')}/commune/config/inference/regression/crypto/sushiswap.yaml"
    def __init__(self,
                 cfg=None):
        super().__init__(cfg=cfg)
        self.cfg = cfg
        self.experiment = ray.get_actor("experiment_manager")
        self.data = ray.get_actor("data_manager")


    def load_models(self, cfg, run, device='cuda'):

        self.model = {}

        with mlflow.start_run(run_id=run.info.run_id) as run:
            for model_key, model_cfg in cfg['model'].items():
                obj_path = f"model.{cfg['model'][model_key]['module']}.CompleteModel"
                model_class =get_object(obj_path)
                self.model[model_key] = model_class(**{'cfg':model_cfg, 'data': self.data})
                self.model[model_key].load()
                self.model[model_key].to(device)
                self.model[model_key].eval()
                self.model[model_key].scale_dropout(scale_factor=cfg['trainer']['eval_dropout_factor'])

    
    def get_inference(self,
                      ### INFERENCE KWARGS
                      tokens= [],
                      timescale= "30m",
                      timestamps=[datetime.datetime.utcnow().timestamp()],
                      periods = {'input': 256, 'output': 48},
                      ### Additional EXPERIMENTS KWARGS
                      experiment=["TEST"],
                      modelName= "nbeats.base",
                      metricName= "val.MSE_future_token0Price",
                      ascending=  True,
                      updatePipeline=False,
                      device="cuda",

                      ):

        
        if not tokens:
            tokens = self.cfg['tokens']

        ray.get(self.data.add_tokens.remote(tokens=tokens,     
                                            run_override_list=['sample_generator'],
                                             update=updatePipeline))

        run, cfg = ray.get(self.experiment.get_best_run.remote(experiment=experiment,
                                                                modelName=modelName,
                                                                metricName=metricName,
                                                                ascending=  ascending))

        self.load_models(run=run, cfg=cfg,device=device)
        batch_dict = ray.get(self.data.get_batch.remote(timescale=timescale, timestamps=timestamps))
        # self.model['oracle'].set(keys="cfg.period", value=periods)
        token_out_dict = self.model['oracle'].evaluate(batch_dict)[timescale]
        for token, sample_dict in token_out_dict.items():
            # generate value from model if it does not exist
            out_dict = {}
            input_period = cfg['data']['pipeline']['periods']['input']

            target_name = cfg['data']['pipeline']['gt_keys'][0]
            out_dict['mean'] = sample_dict[f"pred_future_{target_name}-mean"]
            out_dict['upper'] = sample_dict[f"pred_future_{target_name}-upper"]
            out_dict['lower'] = sample_dict[f"pred_future_{target_name}-lower"]

            out_dict = {k: torch.cumprod(torch.tensor(v+1) , dim=1) for k, v in out_dict.items()}
            out_dict['timestamp'] = torch.tensor(sample_dict[f"timestamp"][:, input_period:])

            
            out_dict['gt'] = sample_dict[f"gt_future_{target_name}-raw"]
            out_dict['gt_past'] = sample_dict[f"gt_past_{target_name}-raw"]
            out_dict['timestamp_past'] = torch.tensor(sample_dict[f"timestamp"][:, :input_period])


            out_dict = {k:v.cpu().numpy()for k,v in out_dict.items()}

            out_dict['timestamp'] =  out_dict['timestamp'].astype(np.int32)
            out_dict['timestamp_past'] =  out_dict['timestamp_past'].astype(np.int32)
            out_dict = {k:[list(v[i]) for i in range(len(v))] for k,v in out_dict.items()}
            token_out_dict[token] = out_dict

        return token_out_dict

    def get(self, item, verbose=False):
        if hasattr(self,item):
            return getattr(self,item)
            if verbose:
                print(f"Succsesfully retrieved {item}")
        else:
            if verbose:
                print(f"{item} does not exist")


    def get_ratios(self, inference_kwargs={},
                   future_steps=3,
                   min_value=0.90,
                   scale=100):

        token_pred_dict = self.get_inference(**inference_kwargs)



        token_score_dict = {}
        timestamp_dict = {}
        future_index = future_steps - 1

        

        for token, pred_dict in token_pred_dict.items():
            sample_count = len(pred_dict['mean'])

            token_score_dict[token] =  scale*torch.tensor(pred_dict['mean']).float()[:,future_index, None]-min_value
            token_score_dict[token] = torch.where(token_score_dict[token]<0, torch.zeros_like(token_score_dict[token]),token_score_dict[token] )
            timestamps = torch.tensor(pred_dict['timestamp'])[:,future_index].tolist()

        print({k:v.shape for k,v in token_score_dict.items()})
        ratio_list = torch.split(torch.softmax(torch.cat(list(token_score_dict.values()), dim=1).float(), dim=1),dim=1, split_size_or_sections=1)
        ratio_list = list(map(lambda x: x.squeeze(1).tolist(), ratio_list))

        output_list = []
        for i in range(sample_count):
            output_list.append(dict(
                ratios=[r[i] for r in ratio_list],
                tokens=list(token_score_dict.keys()),
                timestamps= timestamps,

            ))

        return output_list


    @staticmethod
    def get_swaps(token_prediction_dict,
                  future_steps=3):
        token_dict = {}

        future_index = round(future_minutes / 15)
        for token, pred_dict in token_prediction_dict.items():
            token_dict[token] = {'score': np.array(prediction_dict["mean"])[:, future_index],
                                'future_timestamp': list(np.array(pred_dict["timestamp"])[:,future_index]),
                               'current_timestamp': list(np.array(pred_dict["timestamp"])[:,0])}

        swap_list = []
        for token0, token0_dict in token_dict.items():
            for token1, token1_dict in token_dict.items():
                if token0 != token1:
                    swap_score = list(token0_dict['score'] - token1_dict['score'])
                    swap_list.append(dict(token0=token0,
                                          token1=token1,
                                          score=swap_score,
                                          timescale=timescale,
                                          futureDatetime=list(map(lambda ts: datetime.datetime.utcfromtimestamp(ts).isoformat(),
                                                    token1_dict['future_timestamp'])),
                                          currentDatetime = list(map(lambda ts: datetime.datetime.utcfromtimestamp(ts).isoformat(),
                                               token1_dict['current_timestamp']))
                                          ))

        return swap_list
    
    
    @staticmethod
    def get_dummy_ratios(tokens, start_datetimes=[roundTime(datetime.datetime.utcnow()),roundTime(datetime.datetime.utcnow()) ], future_period_timestamp=3600*3 ):
        ratio_dict_list = []
        for start_datetime in start_datetimes:
            ratio_dict_list.append(dict(
                tokens=tokens,
                ratios= torch.softmax(torch.randn((len(tokens),)), dim=0).tolist(),
                currentDatetime= start_datetime,
                currentTimestamp= start_datetime.timestamp(),
                futureDatetime = datetime.datetime.utcfromtimestamp(start_datetime.timestamp() + future_period_timestamp),
                futureTimestamp =start_datetime.timestamp() + future_period_timestamp
            ))

        return ratio_dict_list


    def test(self):
        current_timestamp= roundTime(datetime.datetime.utcnow()),
        timestamps = [datetime.datetime.utcnow().timestamp() - 3600*h for h in range(10)]
        tokens = ConfigLoader(path=f"{os.getenv('PWD')}/commune/config/config.meta.crypto.tokens/commune_app.yaml", load_config=True).cfg
        tokens = tokens[:]
        inference_kwargs = dict( \
            tokens=tokens,
            timestamps=timestamps,
            timescale="60m",
            experiment=["BOBBEH"],
            modelName="nbeats",
            metricName="val.MSE_future_tokenPriceUSD",
            updatePipeline=False
        )

        results = self.get_ratios(inference_kwargs)
        print(results)
    def __del__(self):
        pass

    
        
if __name__== "__main__":
    experiment_manager = ExperimentManager.deploy(actor=True)
    data_manager = Dataset.deploy(actor=True)
    inference_manager = InferenceManager.deploy(actor=False)
    inference_manager.test()


    


