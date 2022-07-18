"""
`DeepAR: Probabilistic forecasting with autoregressive recurrent networks
<https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_
which is the one of the most popular forecasting algorithms and is often used as a baseline
"""
from copy import copy, deepcopy
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from commune.model.block.nn import get_rnn, MultiEmbedding

class DeepAR(nn.Module):
    def __init__(
        self,
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_features: List[str] = [],
        temporal_features: List[str] = [],
        categorical_features: Dict[str, List[str]] = {},
        known_future_features: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_size = 8,
        embedding_paddings: List[str] = [],
        targets: List[str] = [],
        device = None,
        distribution= None,
        periods: Dict[str,int] = {}
    ):

        # store loss function separately as it is a module
        super().__init__()
        self.__dict__.update(locals())


        self.check_config()

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.embedding_sizes,
            embedding_paddings=self.embedding_paddings,
            max_embedding_size=self.hidden_size,
        )


        self.static_categoricals = [ f for f in self.static_features if f in self.categorical_features]
        self.static_reals = [f for f in self.static_features if f not in self.categorical_features]
        self.temporal_categorical = [f for f in self.temporal_features if f in self.categorical_features]
        self.temporal_reals = [f for f in self.temporal_features if  f not in self.categorical_features]

        self.temporal_categoricals_encoder = self.temporal_categorical
        self.temporal_categoricals_decoder = [f for f in self.temporal_categorical if f in self.known_future_features]

        self.temporal_reals_encoder = self.temporal_reals
        self.temporal_reals_decoder = [f for f in self.temporal_reals if f in self.known_future_features]


        # Build Encoder

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.embedding_sizes[name][1] for name in self.temporal_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: 1
                for name in self.temporal_reals_encoder
            }
        )

        encoder_input_size = sum(encoder_input_sizes.values())
        self.encoder_variables = list(encoder_input_sizes.keys())

        rnn_class = get_rnn(cell_type)
        self.rnn_encoder = rnn_class(
            input_size=encoder_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            dropout=self.dropout if self.rnn_layers > 1 else 0,
            batch_first=True,
        )

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


        decoder_input_size = sum(decoder_input_sizes.values())
        self.decoder_variables = list(decoder_input_sizes.keys())

        rnn_class = get_rnn(cell_type)


        self.rnn_decoder = rnn_class(
            input_size=decoder_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            dropout=self.dropout if self.rnn_layers > 1 else 0,
            batch_first=True,
        )



        self.distribution_projector = nn.ModuleList(
            [nn.Linear(self.hidden_size, len(self.distribution.distribution_arguments)) for args in range(len(self.targets)) ]
        )

    def encode(self, x: torch.Tensor,
               encoder_lengths: torch.Tensor):
        """
        Encode sequence into hidden state
        """
        # encode using rnn
        assert encoder_lengths.min() > 0

        _, hidden_state = self.rnn_encoder(
            x, lengths=encoder_lengths, enforce_sorted=False
        )  # second ouput is not needed (hidden state)
        return hidden_state

    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
        lengths: torch.Tensor = None,
    ):
        decoder_output, hidden_state = self.rnn_decoder(x, hidden_state, lengths=lengths, enforce_sorted=False)



        output = torch.stack([projector(decoder_output) for projector in self.distribution_projector], dim=-1)

        return output, hidden_state

    def decode(
        self,
        x: torch.tensor,
        decoder_lengths: torch.Tensor,
        target_vector : torch.Tensor,
        hidden_state: torch.Tensor):
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """

        if self.training:
            x = torch.cat([x,target_vector], dim=-1)
            output_params, _ = self.decode_all(x, hidden_state, lengths=decoder_lengths)
            output_dist_params = output_params

        else:

            current_hidden_state = hidden_state

            output_dist_params_list = []
            max_decoder_length = int(decoder_lengths.max())


            for idx in range(max_decoder_length):

                # get lagged targets
                x_step = torch.cat([x[:,[idx]], target_vector], dim=-1)
                output_params, current_hidden_state = self.decode_all(
                    x_step,
                    hidden_state=current_hidden_state,
                    lengths= torch.full_like(decoder_lengths, fill_value=1)
                )
                output_dist = self.distribution(output_params.transpose(2, 3))
                target_vector = output_dist.sample(n_samples = 20).mean(0)
                output_dist_params_list.append(output_params)

            output_dist_params = torch.cat(output_dist_params_list, dim=1)



        return output_dist_params



    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """



        x_cont = {k:v if len(v.shape) == 3 else v.unsqueeze(-1)
                  for k,v in x.items()
                  if k in self.temporal_reals}   # concatenate in time dimension
        x_cat = {k:v
                 for k,v in x.items()
                 if k in self.temporal_categorical} # concatenate in time dimension


        input_vectors = self.embeddings(x_cat)
        input_vectors.update(x_cont)

        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]

        max_encoder_length = int(encoder_lengths.max())
        max_decoder_length = int(decoder_lengths.max())


        input_vectors_encoder = torch.cat([v[:, :max_encoder_length,]
                                 for k,v in input_vectors.items()], dim=-1)


        input_vectors_decoder = torch.cat([v[:, max_encoder_length:]
                                 for k,v in input_vectors.items()
                                 if k in self.known_future_features], dim=-1)

        if self.training:

            target_vector = torch.cat([x[target_key][:, max_encoder_length-1:-1]
                                       for target_key in self.targets], dim=-1).unsqueeze(-1)
        else:
            target_vector= torch.cat([x[target_key][:, max_encoder_length:max_encoder_length+1]
                                               for target_key in self.targets], dim=-1).unsqueeze(-1)

        hidden_state = self.encode(x=input_vectors_encoder,
                                   encoder_lengths=encoder_lengths)
        output_dist_params = self.decode(
            x= input_vectors_decoder,
            target_vector=target_vector,
            decoder_lengths=x["decoder_lengths"],
            hidden_state=hidden_state,
        )
        # return relevant part

        return {k:v.squeeze(-1)
                for k,v in
                zip(self.targets,torch.split(output_dist_params, [1]*len(self.targets), dim=-1))}

    def check_config(self):

            assert(all([target in self.temporal_features for target in self.targets]),
                   "target label needs to be in the temporal input featurs")
