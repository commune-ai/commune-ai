import torch
from typing import Dict

def check_tensor_dict_encoder_decoder_shape(tensor_dict: Dict[str, torch.Tensor],
                                            encoder_length: int,
                                            decoder_length: int):
    for k,v in tensor_dict.items():
        assert(v.size(1) == (encoder_length+decoder_length),
               f"Encoder Decoder Mismatch ERROR (KEY: {k.upper()}): "
               f"\n decoder_len: {decoder_length}"
               f"\n encoder_len: {encoder_length}"
               f"\n {k} len : {v.size()} || enc + dec : {encoder_length+decoder_length}")