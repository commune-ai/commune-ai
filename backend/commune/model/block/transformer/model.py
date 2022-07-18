import torch
from torch import nn
from typing import List, Dict, Tuple
from commune.model.block.nn import MultiEmbedding
from .block import \
    EncoderDecoder, \
    Encoder, \
    Decoder, \
    EncoderLayer, \
    DecoderLayer, \
    MultiHeadedAttention, \
    PositionwiseFeedForward,\
    PositionalEncoding,\
    Time2Vec

from copy import deepcopy

class Time2VecTransformer(nn.Module):
    def __init__(self,

                 d_model: int,
                 attn_heads: int,
                 dropout: float,
                 periods: Dict[str,int],
                 d_ff: float,
                 num_layers: int,
                 positional: bool,
                 output_dim: int = 1,
                 temporal_features: List[str] = [],
                 categorical_features: Dict[str, List[str]] = {},
                 known_future_features: List[str] = [],
                 embedding_sizes: Dict[str, Tuple[int, int]] = {},
                 embedding_size=8,
                 embedding_paddings: List[str] = [],
                 targets: List[str] = ['Default'],

                 ):
        self.__dict__.update(locals())
        super(Time2VecTransformer, self).__init__()

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.embedding_sizes,
            embedding_paddings=self.embedding_paddings,
            max_embedding_size=self.embedding_size
        )

        self.temporal_categorical = [f for f in self.temporal_features if f in self.categorical_features]
        self.temporal_reals = [f for f in self.temporal_features if f not in self.categorical_features]

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

        # decoder_input_sizes.update(
        #     {
        #         name: 1
        #         for name in self.targets
        #     }
        # )


        decoder_input_size = sum(decoder_input_sizes.values())
        self.decoder_variables = list(decoder_input_sizes.keys())

        # time to vec layer
        self.t2v_enc_layer = Time2Vec(input_dim=encoder_input_size, k=d_model)
        self.t2v_dec_layer = Time2Vec(input_dim=decoder_input_size+d_model, k=d_model)


        attn_layer = MultiHeadedAttention(attn_heads, d_model)
        self.position_layer = PositionalEncoding(d_model, dropout, periods['input'])
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        # encoder decoder layer for transformer
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, deepcopy(attn_layer) ,deepcopy(ff), dropout), num_layers),
            Decoder(DecoderLayer(d_model, deepcopy(attn_layer),  deepcopy(attn_layer),deepcopy(ff),  dropout), num_layers)
        )

        self.final_layer = nn.ModuleDict({ time_mode: nn.Linear(d_model, output_dim*len(self.targets))
                                           for time_mode in ["past", "future"]})

    def forward(self, x, output_type=Dict):

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


        input_encoder_vector = torch.cat([v[:, :max_encoder_length,]
                                 for k,v in input_vectors.items()], dim=-1)


        input_decoder_vector = torch.cat([v[:, max_encoder_length:]
                                 for k,v in input_vectors.items()
                                 if k in self.known_future_features], dim=-1)

        t2v_enc_emb = self.t2v_enc_layer(input_encoder_vector)

        if self.positional:
            t2v_enc_emb = self.position_layer(t2v_enc_emb)
        enc_emb = self.model.encode(src=t2v_enc_emb)

        in_dec_emb = enc_emb[:,-1:, :]
        dec_emb_list = [in_dec_emb]
        for f in range(self.periods['output']):
            dec_emb_list[-1] = self.t2v_dec_layer(torch.cat([dec_emb_list[-1],
                                                             input_decoder_vector[:,f:f+1,:]], dim=-1))
            in_dec_emb = torch.cat(dec_emb_list, dim=1)
            dec_emb = self.model.decoder(in_dec_emb, enc_emb)
            dec_emb_list.append(dec_emb[:,-1:,:])

        dec_emb = torch.cat(dec_emb_list[1:],dim= 1)

        encoder_output = torch.stack(torch.split(self.final_layer["past"](enc_emb),self.output_dim, dim=-1), dim=-2)
        decoder_output = torch.stack(torch.split(self.final_layer["future"](dec_emb),self.output_dim, dim=-1), dim=-2)


        if output_type == Tuple:
            return encoder_output, decoder_output
        elif output_type == Dict:
            return   {'past': encoder_output,
                      'future': decoder_output}
