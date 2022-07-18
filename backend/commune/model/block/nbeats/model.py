import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from commune.utils.misc import get_object
from typing import Any, Dict, List, Tuple, Union
from commune.model.block.nn import MultiEmbedding
from commune.model.block.transformer.block import \
    EncoderDecoder, \
    Encoder, \
    Decoder, \
    EncoderLayer, \
    DecoderLayer, \
    MultiHeadedAttention, \
    PositionwiseFeedForward,\
    PositionalEncoding,\
    Time2Vec

class NBEATS_Multivariate(nn.Module):
    def __init__(self,
                 output_dim : int = 1,
                 temporal_features: List[str] = [],
                 categorical_features: Dict[str, List[str]] = {},
                 known_future_features: List[str] = [],
                 embedding_sizes: Dict[str, Tuple[int, int]] = {},
                 embedding_size=8,
                 embedding_paddings: List[str] = [],
                 targets: List[str] = [],
                 blocks: List[Dict] = []
                 ):

        ''''

        block_config_list:
        list of block configs
            block: Name of Block module (not included in module kwargs)
            units: number of dimensions
            thetas_dim: number of dimensions for parameters
            num_block_layers: number of layers
            backcast_length: length of past prediction
            forecast_length: length of future prediction,
            dropout: dropout name,

        '''
        super().__init__()
        self.__dict__.update(locals())


        self.embeddings = MultiEmbedding(
            embedding_sizes=self.embedding_sizes,
            embedding_paddings=self.embedding_paddings,
            max_embedding_size=self.embedding_size
        )

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

        decoder_input_size = sum(decoder_input_sizes.values())
        self.decoder_variables = list(decoder_input_sizes.keys())


        self.init_layer = nn.ModuleDict()
        self.init_layer['encoder'] = nn.Linear(encoder_input_size, 1)
        self.init_layer['decoder'] = nn.Linear(decoder_input_size, 1)

        # setup stacks
        block_root = 'model.block.nbeats.sub_modules'
        self.net_blocks = nn.ModuleDict()
        for block_key,block_config in self.blocks.items():
            block_class = get_object(f"{block_root}.{block_config['block']}")
            block_kwargs = {k:v for k,v in block_config.items() if k!='block'}
            self.net_blocks[block_key]=block_class(**block_kwargs)

    def forward(self, x):

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

        backcast_context = self.init_layer['encoder'](input_encoder_vector)
        forcast_context = self.init_layer['decoder'](input_decoder_vector)



        # (batch, sequence, emb_dim) --> (batch, num_targets*output_dim, sequence)
        backcast_context = backcast_context\
                                .expand(-1,-1,len(self.targets)*self.output_dim).contiguous()\
                                .transpose(2,1).contiguous()

        forcast_context = forcast_context\
                                .expand(-1,-1,len(self.targets)*self.output_dim).contiguous()\
                                .transpose(2,1).contiguous()

        backcast = backcast_context
        forcast = forcast_context

        backcast_block_output_dict = {}
        forecast_block_output_dict = {}

        for block_key, block in self.net_blocks.items():
            backcast_block, forcast_block = block(backcast)
            backcast = backcast - backcast_block
            forcast = forcast + forcast_block

            # (batch, num_targets*output_dim, sequence) --> (batch, sequence, num_targets, output_dim)
            backcast_block_output_dict[block_key] = torch.stack(torch.split(backcast_block.transpose(1,2).contiguous(), self.output_dim, dim=-1), dim=-2)
            forecast_block_output_dict[block_key] = torch.stack(torch.split(forcast_block.transpose(1,2).contiguous(), self.output_dim, dim=-1), dim=-2)

        # (batch, num_targets*output_dim, sequence) --> (batch, sequence, num_targets, output_dim)
        backcast = torch.stack(torch.split(backcast.transpose(1,2).contiguous(), self.output_dim, dim=-1), dim=-2)
        forcast = torch.stack(torch.split(forcast.transpose(1,2).contiguous(), self.output_dim, dim=-1), dim=-2)

        return backcast,  forcast



class NBEATS_Time2VecTransformer(nn.Module):
    """
    :param model_args:
          # GRU rnn encoder
          d_model: 16
          d_ff: 16
          dropout: 0.3
          heads: 4
          positional: 1
          N: 1

          periods:
            input: 10
            output: 10

          # we do not use the optimizer in this class (for now tehe)
          optimizer:
            lr: 0.001
            weight_decay: 1.e-4
            amsgrad: True


    """
    def __init__(self, cfg):
        super(Nbeats_Time2VecTransformer, self).__init__()

        self.position_layer = PositionalEncoding(cfg['d_model'], cfg['dropout'], cfg['periods']['input'])
        # time to vec layer
        self.t2v_layer = Time2Vec(input_dim=cfg['input_dim'], k=cfg['d_model'])

        # encoder decoder layer for transformer
        ff = PositionwiseFeedForward(cfg['d_model'], cfg['d_ff'], cfg['dropout'])
        attn_layer = MultiHeadedAttention(cfg['heads'], cfg['d_model'])
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(cfg['d_model'], deepcopy(attn_layer) ,deepcopy(ff), cfg['dropout']), cfg['N']),
            Decoder(DecoderLayer(cfg['d_model'], deepcopy(attn_layer),  deepcopy(attn_layer),deepcopy(ff),  cfg['dropout']), cfg['N'])
        )

        self.final_layer = nn.ModuleDict({ time_mode: nn.Linear(cfg['d_model'], 1) for time_mode in ["past", "future"]})


        if "nbeats" in cfg:
            self.nbeats = NBeat_Regression_Stack_V0(block_list=
                                        [
                                            NBeat_Linear_Block(input_dim_dict = {"input": cfg['input_dim'], "past":cfg['periods']['input'] },
                                                         time_dim_dict={"past": cfg['periods']['input'],
                                                                        "future":cfg['periods']['output']},
                                                          **cfg['nbeats']['linear']),

                                            NBeat_Fourier_Block(input_dim_dict = {"past": cfg['periods']['input'],
                                                                                  "future":cfg['periods']['output']},
                                                             time_dim_dict={"past": cfg['periods']['input'],
                                                                            "future": cfg['periods']['output']},
                                                            **cfg['nbeats']['fourier'])
                                         ]
                                                    )


        self.cfg = cfg

    def forward(self, **kwargs):
        x = torch.stack([v for k,v in kwargs.items() if k in self.cfg['input_columns']], dim=-1)

        t2v_emb = self.t2v_layer(x)
        if self.cfg['positional']:
            t2v_emb = self.position_layer(t2v_emb)

        enc_emb = self.model.encode(src=t2v_emb)

        in_dec_emb = enc_emb[:,-1:, :]
        dec_emb_list = [in_dec_emb]
        for f in range(self.cfg['periods']['output']):
            in_dec_emb = torch.cat(dec_emb_list, dim=1)
            dec_emb = self.model.decoder(in_dec_emb, enc_emb)

            dec_emb_list.append(dec_emb[:,-1:,:])

        dec_emb = torch.cat(dec_emb_list[1:],dim= 1)


        output_dict = {}
        """
        Get Tensors from Transformer :
            - Predict the future and past with the final layer 
        
        """

        output_dict["future"] = self.final_layer["future"](dec_emb).squeeze(-1)
        output_dict["past"] = self.final_layer["past"](enc_emb).squeeze(-1)

        """
        Use Nbeats to Predict on top of the transformer
            - at the moment this would be equivalent to ensembling the transformer with teh nbeats
        """

        if hasattr(self, 'nbeats'):
            pred_nbeats = self.nbeats({"input": x,  "past": output_dict["past"]})
            for time_key in ["past", "future"]:
                output_dict[time_key] = output_dict[time_key]+ pred_nbeats[time_key]


        return output_dict







