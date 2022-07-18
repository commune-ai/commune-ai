from sentence_transformers import SentenceTransformer
import os, sys
if __name__ == '__main__':
    sys.path.append(os.environ['PWD'])

    
from commune.model.base import GenericBaseModel
import torch
import ray
from commune.utils.misc import dict_put



class ModelModule(GenericBaseModel):
    default_cfg_path=os.environ['PWD']+'/'+__file__.replace('.py', '.yaml')
    def __init__(self, cfg):
        GenericBaseModel.__init__(self, cfg)
        self.model = SentenceTransformer(self.cfg['transformer'])
    

    def encode(self, sentences=[]):
        if isinstance(sentences, str):
            sentences = [sentences]

        assert all([isinstance(s, str) for s in sentences]), 'needs ot be all sentences'
        embeddings =  self.forward(sentences)
        return dict(zip(sentences, embeddings))

    def self_similarity(self, sentences=[], output_dict=True):
        embedding_map = self.encode(sentences)
        embeddings = torch.tensor(list(embedding_map.values()))
        
         similarity_matrix = torch.einsum('ij,kj -> ik', embeddings, embeddings).cpu().numpy()

        if output_dict:
            for i, s_i in enumerate(sentences):
                for  j, s_j in enumerate(sentences):
                    dict_put(out_dict, keys=[s_i, s_j], value=similarity_matrix[i][j])

            return out_dict
        else:
            return similarity_matrix

    def forward(self, sentences):
        return self.model.encode(sentences)


if __name__ == '__main__':
    with ray.init(address="auto",namespace="commune"):
        model = ModelModule.deploy(actor={'refresh': False})
        sentences = ['ray.get(model.encode.remote(sentences))', 'ray.get(model.encoder.remote(sentences)) # whadup fam']
        print(ray.get(model.self_similarity.remote(sentences)))

        