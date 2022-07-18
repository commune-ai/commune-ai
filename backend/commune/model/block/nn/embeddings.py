from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class TimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.forward(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = super().forward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class MultiEmbedding(nn.Module):
    def __init__(
        self,
        embedding_sizes: Dict[str, Tuple[int, int]],
        embedding_paddings: List[str],
        max_embedding_size: int = None,
        categorical_groups: Dict[str, List[str]] = {}

    ):
        '''
        Args:
            embedding_sizes: dictionary of tuples of tuple (number_of_categories, category dimension)
            categorical_groups: Dict[str, List[str]],
            embedding_paddings: List[str],
            max_embedding_size: int = None,

        '''


        super().__init__()


        self.embedding_sizes = embedding_sizes
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size

        self.init_embeddings()

    def init_embeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            # convert to list to become mutable
            self.embedding_sizes[name] = list(self.embedding_sizes[name])
            self.embedding_sizes[name][1] = embedding_size
            if name in self.categorical_groups:  # embedding bag if related embeddings
                self.embeddings[name] = TimeDistributedEmbeddingBag(
                    self.embedding_sizes[name][0], embedding_size, mode="sum", batch_first=True
                )
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    padding_idx=padding_idx,
                )

    def names(self):
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str):
        return self.embeddings[name]

    def forward(self, x):
        '''
        Args:
            x: dict(str, torch.Tensor)
        Returns:
            dict(str, torch.Tensor)


        '''
        input_vectors = {}
        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                input_vectors[name] = emb(
                    torch.cat([x[cat_name].long() for cat_name in self.categorical_groups[name]],dim=-1)
                )
            else:
                x[name] = x[name].long()
                input_vectors[name] = emb(x[name])

        return input_vectors

## WIP
class TemporalVariableEncoder(nn.Module):
    def __init__(self,
                 temporal_features: List[str] = [],
                 categorical_features: List[str] = [],
                 known_future_features: List[str] = [],
                 embedding_sizes: Dict[str, Tuple[int, int]] = {},
                 embedding_size=8,
                 embedding_paddings: List[str] = []
                 ):

        """

        """
        self.__dict__.update(locals())

        super().__init__()

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.embedding_sizes,
            embedding_paddings=self.embedding_paddings,
            max_embedding_size=self.embedding_size,
        )

        self.temporal_categorical = [f for f in self.temporal_features if f in self.categorical_features]
        self.temporal_reals = [f for f in self.temporal_features if f not in self.categorical_features]

        # Build Encoder

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.embedding_sizes[name][1] for name in self.categorical_features
        }
        encoder_input_sizes.update(
            {
                name: 1
                for name in self.temporal_reals
            }
        )

        self.encoder_input_size = sum(encoder_input_sizes.values())


        # Build Decoder

        decoder_input_sizes = {
            name: self.embedding_sizes[name][1] for name in self.temporal_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: 1
                for name in self.temporal_reals_decoder
            }
        )

        decoder_input_sizes.update(
            {
                name: 1
                for name in self.targets
            }
        )

        self.decoder_input_size = sum(encoder_input_sizes.values())
        self.decoder_variables = list(decoder_input_sizes.keys())

    def forward(self, x):

        x_cont = {k:v if len(v.shape) == 3 else v.unsqueeze(-1)
                  for k,v in x.items()
                  if k in self.temporal_reals}   # concatenate in time dimension
        x_cat = {k:v
                 for k,v in x.items()
                 if k in self.temporal_categorical} # concatenate in time dimension


        input_vectors = self.embeddings(x_cat)
        input_vectors.update(x_cont)

        return input_vectors

