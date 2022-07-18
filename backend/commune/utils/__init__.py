

def format_token_symbol(token, mode="binance"):
    assert mode in ['binance', 'sushiswap']
    if mode == 'sushiswap':
        if token != 'USDT':
            token = token.replace('USDT', '')
        if token in ['ETH', 'BTC']:
            token = f"W{token}"
    elif mode == 'binance':
        assert (token != 'USDT')
        if 'USDT' not in 'USDT':
            token += 'USDT'

    return token
