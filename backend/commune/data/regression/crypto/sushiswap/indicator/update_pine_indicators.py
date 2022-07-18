import numpy as np
import pandas as pd
from . import pine_functions


def arrayEma(data, n):
    data = np.nan_to_num(data)
    m = (2 / (n + 1))
    f = sum(data[:n]) / n
    emas = [0] * (n - 1)
    emas.append(f)
    for i in range(n, len(data)):
        emas.append((data[i] - emas[i - 1]) * m + (emas[i - 1]))
    return emas


def arraySma(data, n):
    data = np.nan_to_num(data)
    smas = [0] * (n - 1)
    for i in range(n, len(data) + 1):
        smas.append(sum(data[i - n:i]) / n)
    return smas


def rate_of_change(precalculated, calculate, n, src='Close'):
    """
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    ln = len(precalculated)
    concat = pd.Series(list(precalculated[src].loc[ln - n:ln].values) + list(calculate[src].values))

    M = concat.diff(n - 1)
    N = concat.shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    return ROC[len(ROC) - len(calculate):].values


def zvwap(precalculated, calculate, n=20):
    mean = []

    ln = len(precalculated)
    concat_close = pd.Series(list(precalculated['Close'].loc[ln - n * 2:ln].values) + list(calculate['Close'].values))
    concate_vol = pd.Series(list(precalculated['Volume'].loc[ln - n * 2:ln].values) + list(calculate['Volume'].values))

    for i in range(len(concat_close)):
        mean.append((concate_vol.iloc[i - n:i] * concat_close.iloc[i - n:i]).sum() / (concate_vol.iloc[i - n:i]).sum())
    wap = arraySma(concat_close - mean, n)
    zvwap = (concat_close - mean) / wap
    return zvwap.values[n * 2:]


def atr(df, n):
    tr = [df['High'].iloc[0] - df['Low'].iloc[0]]
    for i in range(1, len(df)):
        tr.append(max(df['High'].iloc[i] - df['Low'].iloc[i], abs(df['High'].iloc[i] - df['Close'].iloc[i - 1]),
                      abs(df['Low'].iloc[i] - df['Close'].iloc[i - 1])))
    return arrayEma(tr, n)


def r(precalculated, calculate, n=14):
    """Calculate William's %R for the given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    rs = []
    ln = len(precalculated['High']) - 1
    concat_high = list(precalculated['High'].loc[ln - n + 1:ln].values) + list(calculate['High'].values)
    concat_low = list(precalculated['Low'].loc[ln - n + 1:ln].values) + list(calculate['Low'].values)
    concat_close = list(precalculated['Close'].loc[ln - n + 1:ln].values) + list(calculate['Close'].values)
    for i in range(n, len(concat_close)):
        close = concat_close[i]
        high = max(concat_high[i - n + 1:i])
        low = min(concat_low[i - n + 1:i])
        rs.append((high - close) / (high - low) * -100)
    return rs


def adx(precalculated, calculate, adxlen=7, dilen=5):
    concat_high = pd.Series(list(precalculated['High'].values) + list(calculate['High'].values))
    concat_low = pd.Series(list(precalculated['Low'].values) + list(calculate['Low'].values))
    concat_close = pd.Series(list(precalculated['Close'].values) + list(calculate['Close'].values))
    df = pd.DataFrame()
    df['Close'] = concat_close
    df['Low'] = concat_low
    df['High'] = concat_high
    up = concat_high - concat_high.shift(1)
    up.iloc[0] = 0
    down = concat_low.shift(1) - concat_low
    down.iloc[0] = 0
    plusDM = []
    downDM = []
    ln = len(precalculated)
    for i in range(len(up)):
        plusDM.append(up.iloc[i] if up.iloc[i] > down.iloc[i] and up.iloc[i] > 0 else 0)
        downDM.append(down.iloc[i] if down.iloc[i] > up.iloc[i] and down.iloc[i] > 0 else 0)
    truerange = np.array(atr(df, dilen))
    plus = np.array(arrayEma(plusDM, dilen)) / truerange * 100
    minus = np.array(arrayEma(downDM, dilen)) / truerange * 100
    su = plus + minus
    su[su == 0] = 1
    adx = 100 * np.array(arrayEma(abs(plus - minus) / su, adxlen))
    return adx[ln:]


def tsi(precalculated, calculate, r=25, s=13, src='Close'):
    ln = len(precalculated)
    n = 75
    concat = pd.Series(list(precalculated[src].loc[ln - n + 1:ln].values) + list(calculate[src].values))
    m = (concat.shift(-1) - concat).shift(1).fillna(0)
    tsi = 100 * (np.array(pine_functions.ema(pine_functions.ema(m, r), s)) / np.array(
        pine_functions.ema(pine_functions.ema(abs(m), r), s)))
    return tsi[len(concat) - len(calculate):]


def wave_trend(precalculated, calculate, channel_len=10, average_len=10, n=150):
    ln = len(precalculated['High']) - 1
    concat_high = np.array(list(precalculated['High'].loc[ln - n + 1:ln].values) + list(calculate['High'].values))
    concat_low = np.array(list(precalculated['Low'].loc[ln - n + 1:ln].values) + list(calculate['Low'].values))
    concat_close = np.array(list(precalculated['Close'].loc[ln - n + 1:ln].values) + list(calculate['Close'].values))
    ap = (concat_high + concat_low + concat_close) / 3
    esa = np.array(pine_functions.pine_ema(ap, channel_len))
    d = np.array(pine_functions.pine_ema(abs(ap - esa), channel_len))
    ci = (ap - esa) / (0.015 * d)
    tci = np.array(pine_functions.pine_ema(ci, average_len))
    return tci[n:]


def pD(precalculated, calculate, n=190):
    calculate['High'].round(decimals=4)
    calculate['Low'].round(decimals=4)
    calculate['Close'].round(decimals=4)
    calculate['Open'].round(decimals=4)

    ln = len(precalculated['High']) - 1
    coconcat_high = np.array(list(precalculated['High'].loc[ln - n + 1:ln].values) + list(calculate['High'].values))
    concat_low = np.array(list(precalculated['Low'].loc[ln - n + 1:ln].values) + list(calculate['Low'].values))
    concat_close = np.array(list(precalculated['Close'].loc[ln - n + 1:ln].values) + list(calculate['Close'].values))
    coconcat_open = np.array(list(precalculated['Open'].loc[ln - n + 1:ln].values) + list(calculate['Open'].values))
    haclose = (concat_close + coconcat_high + coconcat_open + concat_low) / 4
    haopen = [(coconcat_open[0] + concat_close[0]) / 2]
    hahigh = [max(haclose[0], haopen[0], coconcat_high[0])]
    halow = [min(haclose[0], haopen[0], concat_low[0])]
    for i in range(1, len(concat_close)):
        haopen.append((haopen[i - 1] + haclose[i - 1]) / 2)
        hahigh.append(max(haclose[i], haopen[i], coconcat_high[i]))
        halow.append(min(haclose[i], haopen[i], concat_low[i]))

    src = haclose
    c = np.array(haclose)
    # KJD
    ilong = 9
    isig = 3

    def bcwsma(s, l, m):
        _bcwsma = [(m * s[0]) / l]
        for i in range(1, len(s)):
            _bcwsma.append(((m * s[i]) + (l - m) * _bcwsma[i - 1]) / l)
        return _bcwsma

    h = np.array(pine_functions.highest(hahigh, ilong))
    l = np.array(pine_functions.lowest(halow, ilong))

    RSV = 100 * ((c - l) / (h - l))
    RSV = np.nan_to_num(RSV)

    pK = np.array(bcwsma(RSV, isig, 1))
    pD = np.array(bcwsma(pK, isig, 1))
    return pD[n:]
