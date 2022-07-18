from commune.model.block.gp.block import MultivariateBatchedGP
from commune.model.block.nn import MultiEmbedding
from commune.model.block.transformer.block import MultiHeadedAttention, Time2Vec
from typing import Any, Dict, List, Tuple, Union
from commune.utils.misc import get_object
import torch
import torch.nn as nn
from gpytorch.mlls import ExactMarginalLogLikelihood
from .block import MultivariateBatchedGP



class RegressionGP(nn.Module):
    def __init__(self,

                 output_dim: int,
                 batch_size: int,
                 device: str,
                 use_ard=False,
                 optimizer: Dict = {},
                 num_training_steps=100,
                 reset_training_state=True,
                 periods={},
                 targets: List[str] = [],

                 temporal_features: List[str] = [],
                 categorical_features: Dict[str, List[str]] = {},
                 embedding_sizes: Dict[str, Tuple[int, int]] = {},
                 embedding_size=8,
                 embedding_paddings: List[str] = [],



    ):

        self.__dict__.update(locals())

        super().__init__()

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.embedding_sizes,
            embedding_paddings=self.embedding_paddings,
            max_embedding_size=self.embedding_size,
        )


        self.temporal_categorical = [f for f in self.temporal_features if f in self.categorical_features]
        self.temporal_reals = [f for f in self.temporal_features if  f not in self.categorical_features]

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

        self.input_size = sum(encoder_input_sizes.values())


        self.tim2vec_layer = Time2Vec(input_dim=self.input_size,
                                      k=self.input_size)

        self.tim2vec_layer = nn.Linear(self.input_size, self.input_size)

        self.optimizer_kwargs = optimizer
        self.create_model()

    def create_model(self):

        self.model = MultivariateBatchedGP(input_dim=self.input_size,
                                           output_dim=self.output_dim*len(self.targets),
                                           batch_size=self.batch_size,
                                           device=self.device,
                                           periods=self.periods,
                                           use_ard = False)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        self.optimizer = torch.optim.Adam([{'params': self.parameters()}],
                                            **self.optimizer_kwargs)


    def fit(self, x, y):
        if self.reset_training_state:
            self.model.reset_model_state()

        self.model.set_train_data(inputs=x, targets=y, strict=False)
        self.model.train()

        x_input = x.clone()

        with torch.enable_grad():
            for i in range(self.num_training_steps):

                self.optimizer.zero_grad()
                output = self.model(x_input)
                loss = -self.mll(output, y).mean()
                loss.backward(retain_graph=True)
                self.optimizer.step()

    def predict(self, x):
        out_dict = {}
        self.model.eval()
        with torch.no_grad():
            pred = self.model.likelihood(self.model(x))
            out_dict["mean"] = pred.mean
            out_dict["lower"], out_dict["upper"] = pred.confidence_region()

        return out_dict

    def encode(self, x):
        x_cont = {k: v if len(v.shape) == 3 else v.unsqueeze(-1)
                  for k, v in x.items()
                  if k in self.temporal_reals}  # concatenate in time dimension
        x_cat = {k: v
                 for k, v in x.items()
                 if k in self.temporal_categorical}  # concatenate in time dimension

        input_vectors = self.embeddings(x_cat)
        input_vectors.update(x_cont)

        max_encoder_length = int(x["encoder_lengths"].max())

        emb_vector = torch.cat(list(input_vectors.values()), dim=-1)

        emb_train = emb_vector[:, :max_encoder_length]
        emb_test = emb_vector[:, max_encoder_length:]

        return emb_train, emb_test




    def forward(self, x):

        max_encoder_length = int(x["encoder_lengths"].max())
        y_train = torch.cat([x[k][:, :max_encoder_length] for k in self.targets], 0)
        x_train, x_test = self.encode(x)

        batch_size = x_train.shape[0]//(len(self.targets)*self.output_dim)

        self.batch_size = batch_size
        self.create_model()
        self.to(device=self.device)


        self.fit(x=x_train, y=y_train)
        out_dict = self.predict(x_test)

        self.predict(x_test)

        for p in self.parameters():
            del p
        del self.model

        return out_dict


class RegressionEncoderGP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 batch_size: int,
                 device: str,
                 use_ard=False,
                 num_training_steps=10,
                 reset_training_state=False,
                 periods = {},
                 optimizer = {},
                 targets = []
                 ):
        self.__dict__.update(locals())

        super().__init__()
        self.optimizer_kwargs = optimizer
        self.create_model()


    def create_model(self):
        self.model = MultivariateBatchedGP(
            input_dim=self.input_dim,
            output_dim=self.output_dim * len(self.targets),
            batch_size=self.batch_size,
            device=self.device,
            periods=self.periods,
            use_ard=False

        )

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        self.optimizer = torch.optim.Adam([{'params': self.parameters()}],
                                          **self.optimizer_kwargs)


    def fit(self, x, y):
        if self.reset_training_state:
            self.model.reset_model_state()


        self.model.set_train_data(inputs=x, targets=y, strict=False)
        self.model.train()

        x_input = x.clone()

        with torch.enable_grad():
            for i in range(self.num_training_steps):

                self.optimizer.zero_grad()
                output = self.model(x_input)
                loss = -self.mll(output, y).mean()
                loss.backward(retain_graph=True)
                self.optimizer.step()

    def predict(self, x):
        out_dict = {}
        self.model.eval()
        with torch.no_grad():
            pred = self.model.likelihood(self.model(x))
            out_dict["mean"] = pred.mean
            out_dict["lower"], out_dict["upper"] = pred.confidence_region()

        return out_dict

    def forward(self, x_train, y_train, x_test):
        '''
        x_train: (Y_dim*Batch_dim, Number_samples, input_dim)
        x_test: (Y_dim*Batch_dim, Number_samples, input_dim)
        y_train: (Y_dim*Batch_dim, Number_samples, input_dim)

        '''
        batch_size = x_train.shape[0] // (len(self.targets) * self.output_dim)


        self.batch_size = batch_size
        self.create_model()
        self.to(device=self.device)

        train_period_fraction = 0.2

        x_train = x_train[:,-int(x_train.shape[1]*train_period_fraction):]
        y_train = y_train[:,-int(x_train.shape[1]*train_period_fraction):]
        self.fit(x=x_train, y=y_train)

        pred_train = self.predict(x_train)
        pred_test = self.predict(x_test)

        for p in self.parameters():
            del p
        del self.model



        return pred_train, pred_test