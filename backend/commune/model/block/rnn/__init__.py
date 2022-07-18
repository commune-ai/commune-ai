
import torch
from torch import nn
import random
from typing import List, Dict, Tuple
from commune.model.block.nn import MultiEmbedding

class AttentionRNN(nn.Module):
    """
    LSTM model that takes in a sequence as input and predicts the number of times
    the patient will visit the hospital in the nect year. So output is a real-number (using ReLU)
    """

    def __init__(self,
                 encoder_bidirectional: bool,
                 encoder_mh_attn_heads: int,
                 decoder_mh_attn_heads: int,
                 rnn_hidden_size: int,
                 final_hidden_size:int,
                 dropout_rnn: float,
                 dropout_final: float,
                 final_num_layers: int,
                 periods: Dict[str, int],
                 infuser_attn: bool,
                 teacher_prob: float,
                 predict_past: ["ENC"],

                 output_dim: int = 1,
                 temporal_features: List[str] = [],
                 categorical_features: Dict[str, List[str]] = {},
                 known_future_features: List[str] = [],
                 embedding_sizes: Dict[str, Tuple[int, int]] = {},
                 embedding_size=8,
                 embedding_paddings: List[str] = [],
                 targets: List[str] = [],

                 ):

        encoder_bidirectional_factor = 2 if encoder_bidirectional else 1
        self.__dict__.update(locals())
        super().__init__()

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

        decoder_input_size = sum(decoder_input_sizes.values())
        self.decoder_variables = list(decoder_input_sizes.keys())



        """RNN ENCODER"""

        self.rnn_enc = nn.GRU(
            input_size= encoder_input_size,
            hidden_size=rnn_hidden_size,
            batch_first=True,
            dropout=dropout_rnn,
            bidirectional=encoder_bidirectional
        )

        """RNN DECODER"""
        self.rnn_dec = nn.GRU(
            input_size=decoder_input_size + output_dim*len(targets),
            hidden_size=rnn_hidden_size,
            batch_first=True,
            dropout=dropout_rnn,
            bidirectional=False
        )

        """MULTIHEAD ATTENTION AFTER ENCODER WITH LAYER NORM"""

        if encoder_mh_attn_heads > 0:
            assert rnn_hidden_size % encoder_mh_attn_heads == 0, "MultiHeaded-Attention-Num Heads Aint Right"
            self.encoder_mh_attn = nn.MultiheadAttention(rnn_hidden_size * encoder_bidirectional_factor , encoder_mh_attn_heads)

            self.encoder_mh_attn_alpha = nn.Parameter(torch.empty(1))
            self.encoder_mh_ln = nn.LayerNorm(rnn_hidden_size * encoder_bidirectional_factor)


        """INFUSION DOT ATTENTION BETWEEN FUTURE ELEMENT AND EVERY PAST ENCODING"""
        if infuser_attn:
            self.infuser_attn_ln = nn.LayerNorm(rnn_hidden_size)
            self.infuser_attn = nn.Linear(rnn_hidden_size + (output_dim*len(targets)) + decoder_input_size, periods['input'])
            self.infuser_attn_combine = nn.Linear(rnn_hidden_size + (encoder_bidirectional_factor * rnn_hidden_size), rnn_hidden_size)


        """DECODER MULTIHEADED ATTENTION"""
        if decoder_mh_attn_heads > 0:
            assert rnn_hidden_size % decoder_mh_attn_heads == 0, "MultiHeaded-Attention-Num Heads Aint Right"
            self.decoder_mh_attn = nn.MultiheadAttention(rnn_hidden_size, decoder_mh_attn_heads)
            self.decoder_mh_attn_alpha = nn.Parameter(torch.empty(1))
            self.decoder_mh_ln = nn.LayerNorm(rnn_hidden_size)

            """FINAL SET OF LINEAR LAYERS"""
            self.h_mh_dec = nn.ModuleList([])
            self.ln_mh_dec = nn.ModuleList([])
            for i in range(final_num_layers + 1):
                if i == 0:
                    in_dim_h = rnn_hidden_size
                elif i > 0:
                    in_dim_h = out_dim_h
                out_dim_h = final_hidden_size // (2 ** (i + 1))

                self.h_mh_dec.append(nn.Linear(in_dim_h, out_dim_h, bias=True))
                self.ln_mh_dec.append(nn.LayerNorm(out_dim_h))

            self.out_c_mh_dec = nn.Linear(out_dim_h, output_dim*len(targets), bias=True)



        self.act = nn.ELU()




        """FINAL SET OF LINEAR LAYERS"""
        self.h = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        for i in range(final_num_layers+ 1):
            if i == 0:
                in_dim_h = rnn_hidden_size
            elif i > 0:
                in_dim_h = out_dim_h
            out_dim_h = final_hidden_size // (2 ** (i + 1))

            self.h.append(nn.Linear(in_dim_h, out_dim_h, bias=True))
            self.ln.append(nn.LayerNorm(out_dim_h))

        self.dropout_final = nn.Dropout(dropout_final)


        # OUTPUT LAYER
        self.out_c = nn.Linear(out_dim_h, output_dim*len(targets), bias=True)


        """PREDICTING THE PAST """
        # predicting the past with encoder or decoder
        if self.predict_past in ["ENC", "DEC"]:
            self.h_past = nn.ModuleList([])
            self.ln_past = nn.ModuleList([])

            for i in range(final_num_layers + 1):
                if i == 0:
                    if self.predict_past == "ENC":
                        in_dim_h = self.rnn_hidden_size * encoder_bidirectional_factor
                    elif self.predict_past == "DEC":
                        in_dim_h = self.rnn_hidden_size
                elif i > 0:
                    in_dim_h = out_dim_h
                out_dim_h = self.final_hidden_size // (2 ** (i + 1))

                self.h_past.append(nn.Linear(in_dim_h, out_dim_h, bias=True))
                self.ln_past.append(nn.LayerNorm(out_dim_h))

            self.dropout_final_past = nn.Dropout(dropout_final)
            self.out_c_past = nn.Linear(out_dim_h, output_dim*len(targets), bias=True)



    def forward_predict_past(self, z, pivot):
        # predict the past

        for i in range(len(self.h_past)):
            z = self.dropout_final(self.act(self.ln_past[i](self.h_past[i](z))))


        return self.out_c_past(z) + pivot



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


        out_dict = {}
        # we replicate the scalar features across every time-step


        rnn_outs, h_enc = self.rnn_enc(input_encoder_vector)

        rnn_outs = rnn_outs.contiguous().view(rnn_outs.shape[0], max_encoder_length, self.rnn_hidden_size *(2 if self.encoder_bidirectional else 1))


        # shift the future gt backwards by 1 to include the last input value

        # get the pivot and

        # take a mean across both directions
        if self.encoder_bidirectional:
            h_enc = h_enc.mean(dim=0, keepdim=True)


        h_dec = h_enc

        if self.encoder_mh_attn_heads:
            rnn_out_attn, _ = self.encoder_mh_attn(query=rnn_outs.permute(1, 0, 2),
                                           key=rnn_outs.permute(1, 0, 2),
                                           value=rnn_outs.permute(1, 0, 2))

            # LAYER NORMALIZATION
            rnn_outs = self.encoder_mh_ln(rnn_out_attn.transpose(0,1) + rnn_outs)


        # ground truth is the future column
        target_tensors = torch.cat([input_vectors[k][:, max_encoder_length-1:-1]  for k in self.targets], dim=-1)
        pivot  = target_tensors[:, 0, None]


        rnn_out_dec_list = []
        out_dec_list = []
        out_dec_diff_list = []

        for f in range(max_decoder_length):

            teacher_force = (random.uniform(0,1)> self.teacher_prob)
            if (teacher_force and self.training) or f == 0:
                in_dec = target_tensors[:, f, None, :] - pivot
            else:
                in_dec = out_dec_diff

            in_dec = torch.cat([in_dec, input_decoder_vector[:,f,None] ],dim=-1)

            if self.infuser_attn:
                infuser_attn_weights = nn.Softmax(dim=-1)(self.infuser_attn(torch.cat([h_dec.squeeze(0), in_dec.squeeze(1)], -1)))

                """
                - attn_weights (BATCH_SIZE, MAX_SEQUENCE, MAX_SEQUENCE)
                - rnn_outs (B)
                """

                infuser_attn_applied = torch.bmm(infuser_attn_weights.unsqueeze(1), rnn_outs).permute(1, 0, 2)
                h_dec = self.infuser_attn_ln(self.infuser_attn_combine(torch.cat([infuser_attn_applied, h_dec], -1)) + h_dec)

            rnn_out_dec, h_dec = self.rnn_dec(in_dec,h_dec)
            rnn_out_dec_list.append(rnn_out_dec)


            z = rnn_out_dec
            for i in range(len(self.h)):
                z = self.dropout_final(self.act(self.ln[i](self.h[i](z))))

            out_dec_diff = self.out_c(z)
            out_dec = out_dec_diff + pivot

            out_dec_list.append(out_dec)
            #out_dec_diff_list.append(out_dec_diff)
        rnn_out_dec = torch.cat(rnn_out_dec_list, dim=1)
        out_dec = torch.cat(out_dec_list, dim=1)
        #out_dec_diff = torch.cat(out_dec_diff_list, dim=1)

        # predict past from decoder outputs if args.model.predict_past has "decoder" or "encoder"

        out_dict = {}
        if self.predict_past == "ENC":
            out_dict['past'] = self.forward_predict_past(z=rnn_outs, pivot=pivot)
        elif self.predict_past == "DEC":
            out_dict['past'] = self.forward_predict_past(z=rnn_out_dec, pivot=pivot)


        if self.decoder_mh_attn_heads:

            decoder_attn, _ = self.decoder_mh_attn(query=rnn_out_dec.permute(1, 0, 2),
                                                   key=rnn_out_dec.permute(1, 0, 2),
                                                   value=rnn_out_dec.permute(1, 0, 2))

            # LAYER NORMALIZATION
            rnn_out_dec = self.decoder_mh_ln(decoder_attn.transpose(0, 1) + rnn_out_dec)

            z = rnn_out_dec
            for i in range(len(self.h_mh_dec)):
                z = self.dropout_final(self.act(self.ln_mh_dec[i](self.h_mh_dec[i](z))))
            dec_attn_out_dec = self.out_c_mh_dec(z) + pivot

            out_dec = out_dec + dec_attn_out_dec

        out_dict['future'] = out_dec

        out_dict = {k:torch.stack(torch.split(v,self.output_dim, dim=-1), dim=-2) for k,v in out_dict.items()}

        return out_dict