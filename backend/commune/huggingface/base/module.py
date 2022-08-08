
import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import transformers
import 
from typing import Union

from commune.process.base import BaseProcess
import torch
import ray
from commune.utils.misc import dict_put

class BaseHuggingFaceModule(BaseProcess):
    default_cfg_path= 'commune.huggingface.base.module'

    def __init__(self, cfg=None):
        BaseProcess.__init__(self, cfg=cfg)
        self.load()


    def load(self):
        self.load_dataset()
        self.load_tokenizer()
        self.load_model()
        self.load_metrics()
        self.load_trainer()
    
    def compute_metrics(self,eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def load_tokenizer(self, *args, **kwargs):
        '''load tokenizer'''
        if len(args) + len(kwargs) == 0:
    
            args  = self.cfg.get('tokenizer')
            assert args != None
            if isinstance(args, str):
                args = [args]
                kwargs = {}
            elif type(args) in [list]:
                args = args
                kwargs = {}
            elif type(args) in [dict]:
                kwargs = args
                args = []
            self.tokenizer = transformer.AutoTokenizer(*args, **kwargs)
        else:
            raise NotImplementedError


    def train():
        self.trainer.train()


    def load_model(self):
        model_args = self.cfg['model']
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained()


    def preprocess(self):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)
        # split data
        self.splits = {}
        self.splits['train'] = self.dataset["train"].shuffle(seed=42).select(range(1000))
        self.splits['test'] = self.dataset["test"].shuffle(seed=42).select(range(1000))


    def load_trainer(self):
        trainer_args = self.cfg.get('trainer')
        if trainer_args is None:
            return 

        training_args = transformer.TrainingArguments(args = trainer_args)

        self.trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=self.split['train'],
                    eval_dataset=self.split['test'],
                    compute_metrics=compute_metrics)
)


    '''
    Dataset
    '''


    '''
    Dataset.Metrics
    '''
    def load_metric(self,metric:str, cache=True, *args, **kwargs):
        '''
        Load Metric
        '''
        metric_object = dataset.load_metric(metric, *args, **kwargs)
        if store:
            self.metrics[metric] = metric_object
        return metric_object

    def load_metrics(self, metrics:Union(dict[str, dict], list[str])= None,*args, **kwargs):
        '''
        Multiple Metrics

        '''

        if metrics is None:
            metrics  = self.cfg.get('metric')

        if isinstance(metics: str):
            metrics = [metrics]

        m_index_list = []
        if isinstance(m, list):
            m_index_list = list(range(len(m)))
        elif isinstance(m, dict):
            m_index_list = list(m.keys())
        else:
            raise Exception('metrics should be a list or a dictionar') 
        

        assert len(m_index_list) > 0, f'Bro, the m_index_list is empty, metrics: {metrics}'
        for m_idx in m_index_list:
            
            m_obj = metrics[m_idx]

            m_args = []
            m_kwargs =  {}

            if isinstance(m_obj, list)
                m_args = m_kwargs
            elif isinstance(m_obj, dict):
                m_kwargs = m_obj
            elif isinstance(m_obj, str):
                m_kwargs  = dict(metric=m)

            self.load_metric(*m_args, **m_kwargs)
              
        return self.metrics
    # def encode(self, sentences=[]):
    #     if isinstance(sentences, str):
    #         sentences = [sentences]

    #     assert all([isinstance(s, str) for s in sentences]), 'needs ot be all sentences'
    #     embeddings =  self.forward(sentences)
    #     return dict(zip(sentences, embeddings))

    # def self_similarity(self, sentences=[], output_dict=True):
    #     embedding_map = self.encode(sentences)
    #     embeddings = torch.tensor(list(embedding_map.values()))
        
    #      similarity_matrix = torch.einsum('ij,kj -> ik', embeddings, embeddings).cpu().numpy()

    #     if output_dict:
    #         for i, s_i in enumerate(sentences):
    #             for  j, s_j in enumerate(sentences):
    #                 dict_put(out_dict, keys=[s_i, s_j], value=similarity_matrix[i][j])

    #         return out_dict
    #     else:
    #         return similarity_matrix

    # def forward(self, sentences):
    #     return self.model.encode(sentences)


if __name__ == '__main__':
    with ray.init(address="auto",namespace="commune"):
        model = ModelModule.deploy(actor={'refresh': False})
        sentences = ['ray.get(model.encode.remote(sentences))', 'ray.get(model.encoder.remote(sentences)) # whadup fam']
        print(ray.get(model.self_similarity.remote(sentences)))

        