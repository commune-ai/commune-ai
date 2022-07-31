import os
import ray
from .schema.token import CoinData, SingleCoinSingleTimeScaleData
from commune.config import ConfigLoader
import datetime

def get_token_list(mode = "commune_app"):
    return ConfigLoader(f"{os.getenv('PWD')}/commune/config/config.meta.crypto.tokens/{mode}.yaml").cfg

@ray.remote(num_cpus=1)

def get_token_data(token,
                  timestampMax,
                  timestampMin):

    client = get_client('postgres')
    token = token.lower()
    if 'usdt' not in token:
        token = token + 'usdt'

    token_data_dict = {}
    df = client.query(f'''
        SELECT * FROM extract_crypto_binance_{token} 
        WHERE "Open Time" BETWEEN {timestamp_bound[0]} AND {timestamp_bound[1]}
    
    ''',output_pandas = True)

    df['Open DateTime'] = df['Open Time'].apply(lambda ts: datetime.datetime.fromtimestamp(ts))

    return {'token': token,
            'timescale': timescale,
            'data':dict(
                        Open=df['Open'].tolist(),
                        Low=df['Low'].tolist(),
                        High=df['High'].tolist(),
                        Close=df['Close'].tolist(),
                        Volume=df['Volume'].tolist(),
                        Time=df['Open DateTime'].tolist(),
                        Timestamp=df['Open Time'].tolist()
                        )
            }

def simple_get_signal( mean_signal_list_batch,
               hodl_bounds=[0.999, 1.001]):

    output_signal = []
    for mean_signal_list in mean_signal_list_batch:
        output_signal.append([])
        for v in mean_signal_list:

            if v < hodl_bounds[0]:
                signal= 'SELL'
            elif v >= hodl_bounds[0] and v <= hodl_bounds[0]:
                signal= 'HODL'
            elif v > hodl_bounds[1]:
                signal= 'BUY'
            else:
                signal= 'HODL'

            output_signal[-1].append(signal)
    return output_signal

