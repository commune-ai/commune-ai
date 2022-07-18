"""
The temporal fusion transformer is a powerful predictive model for forecasting timeseries
"""
from copy import copy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn

from commune.model.block.nn import get_rnn, MultiEmbedding
from commune.model.block.temporal_fusion_transformer.sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)
from commune.model.block.temporal_fusion_transformer.utils import create_mask
class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: Union[int, List[int]] = 1,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_features: List[str] = [],
        temporal_features=[],
        categorical_features: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        known_future_features: List[str] = [],
        hidden_continuous_size: int = 8,
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_size= 8,
        embedding_paddings: List[str] = [],
        share_single_variable_networks: bool = False,
        device = None,

        targets= List[str],
        cell_type='LSTM'
    ):
        """
        Temporal Fusion Transformer for forecasting timeseries - use its :py:meth:`~from_data` method if possible.

        Implementation of the article
        `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
        Forecasting <https://arxiv.org/pdf/1912.09363.pdf>`_. The network outperforms DeepAR by Amazon by 36-69%
        in benchmarks.

        Enhancements compared to the original implementation (apart from capabilities added through base model
        such as monotone constraints):

        * static variables can be continuous
        * multiple categorical variables can be summarized with an EmbeddingBag
        * variable encoder and decoder length by sample
        * categorical embeddings are not transformed by variable selection network (because it is a redundant operation)
        * variable dimension in variable selection network are scaled up via linear interpolation to reduce
          number of parameters
        * non-linear variable processing in variable selection network can be shared among decoder and encoder
          (not shared by default)

        Tune its hyperparameters with
        :py:func:`~pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters`.

        Args:

            hidden_size: hidden size of network which is its main hyperparameter and can range from 8 to 512
            lstm_layers: number of LSTM layers (2 is mostly optimal)
            dropout: dropout rate
            output_size: number of outputs (e.g. number of quantiles for QuantileLoss and one target or list
                of output sizes).
            loss: loss function taking prediction and targets
            attention_head_size: number of attention heads (4 is a good default)
            max_encoder_length: length to encode (can be far longer than the decoder length but does not have to be)

            static_reals: names of static
            temporal_features: names of temporal_features
            categorical_features: names of categorical variables (static and temporal)

            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_size: default to embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
                       reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
        """
        # store loss function separately as it is a module

        super().__init__()


        self.__dict__.update(locals())



        # processing inputs
        # embeddings




        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.embedding_sizes,
            categorical_groups=self.categorical_groups,
            embedding_paddings=self.embedding_paddings,
            max_embedding_size=self.hidden_size,
        )

        self.static_categoricals = [ f for f in self.static_features if f in self.categorical_features]
        self.static_reals = [f for f in self.static_features if f not in self.categorical_features]
        self.temporal_categorical = [f for f in self.temporal_features if f in self.categorical_features]
        self.temporal_reals = [f for f in self.temporal_features if  f not in self.categorical_features]

        self.reals = self.static_reals + self.temporal_reals
        self.temporal_categoricals_encoder = self.temporal_categorical


        self.temporal_categoricals_encoder = self.temporal_categorical
        self.temporal_categoricals_decoder = [f for f in self.temporal_categorical if f in self.known_future_features]

        self.temporal_reals_encoder = self.temporal_reals
        self.temporal_reals_decoder = [f for f in self.temporal_reals if f in self.known_future_features]

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hidden_continuous_size)
                for name in self.reals
            }
        )



        # variable selection
        # variable selection for static variables
        static_input_sizes = {name: self.embedding_sizes[name][1] for name in self.static_categoricals}
        static_input_sizes.update(
            {
                name: self.hidden_continuous_size
                for name in self.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={name: True for name in self.static_categoricals},
            dropout=self.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.embedding_sizes[name][1] for name in self.temporal_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hidden_continuous_size
                for name in self.temporal_reals_encoder
            }
        )

        self.encoder_variables = list(encoder_input_sizes.keys())

        decoder_input_sizes = {
            name: self.embedding_sizes[name][1] for name in self.temporal_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hidden_continuous_size
                for name in self.temporal_reals_decoder
            }
        )

        self.decoder_variables = list(decoder_input_sizes.keys())

        # create single variable grns that are shared across decoder and encoder
        if self.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    self.hidden_size,
                    self.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hidden_size),
                        self.hidden_size,
                        self.dropout,
                    )




        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={name: True for name in self.temporal_categoricals_encoder},
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={name: True for name in self.temporal_categoricals_decoder},
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, self.dropout
        )

        # lstm encoder (history) and decoder (future) for local processing

        RNN = get_rnn(self.cell_type)

        self.lstm_encoder = RNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = RNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(self.hidden_size, dropout=self.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(self.hidden_size, dropout=self.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(self.hidden_size, trainable_add=False)
        # self.post_lstm_add_norm_decoder = AddNorm(self.hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hidden_size, n_head=self.attention_head_size, dropout=self.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(
            self.hidden_size, dropout=self.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, dropout=self.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(self.hidden_size, dropout=None, trainable_add=False)
        self.output_layer = nn.ModuleDict()
        for target in self.targets:
            self.output_layer[target] = nn.Linear(self.hidden_size, self.output_size)


    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_length: int):
        """
        Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask


    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]


        max_encoder_length = int(encoder_lengths.max())
        max_decoder_length =int(decoder_lengths.max())

        timesteps = max_encoder_length + max_decoder_length

        x_cat = {k:v for k,v in x.items() if k in self.categorical_features} # concatenate in time dimension
        x_cont = {k:v if len(v.shape) == 3 else v.unsqueeze(-1) for k,v in x.items() if k not in self.categorical_features}   # concatenate in time dimension


        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(x_cont)

        batch_size = list(input_vectors.values())[0].shape[0]
        device = list(input_vectors.values())[0].device

        # Embedding and variable selection
        if len(self.static_features) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_features}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:

            static_embedding = torch.zeros(
                (batch_size, self.hidden_size), device=device
            ).float()
            static_variable_selection = torch.zeros((batch_size, 0), device=device).float()

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }


        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.lstm_layers, -1, -1)

        # run local encoder


        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )



        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=timesteps - max_encoder_length
            ),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output_future = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])

        output_dict = {}
        for target in self.targets:
            output_dict[f'pred_future_{target}'] = self.output_layer[target](output_future)

        return output_dict

