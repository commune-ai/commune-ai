#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:45:50 2019

@author: devinreich
"""
# Import Third-Party
import pandas as pd
import numpy as np
import math
from . import pine_functions

# Divergences
def _top_fractal(src):
    # f_top_fractal(src) => src[4] < src[2] and src[3] < src[2] and src[2] > src[1] and src[2] > src[0]
    return None


def _bot_fractal(src):
    # f_bot_fractal(src) => src[4] > src[2] and src[3] > src[2] and src[2] < src[1] and src[2] < src[0]
    return None


def zvwap(df, n=20):
    mean = [0] * (n)
    for i in range(n, len(df)):
        mean.append((df['Volume'].iloc[i - n:i] * df['Close'].iloc[i - n:i]).sum() / (df['Volume'].iloc[i - n:i]).sum())
    wap =arraySma(df['Close'] - mean, n)
    df['zvwap'] = (df['Close'] - mean) / wap


def _fractalize(src):
    # f_fractalize(src) => f_top_fractal(src) ? 1 : f_bot_fractal(src) ? -1 : 0
    return 1 if _top_fractal(src) else (-1 if _bot_fractal(src) else 0)


def _wavetrend(df, src, chlen, avg, malen):
    tfsrc = df[src]
    esa = np.array(pine_functions.pine_ema(tfsrc, chlen))
    de = np.array(pine_functions.pine_ema(abs(tfsrc - esa), chlen))
    ci = (tfsrc - esa) / (0.015 * de)
    wt1 = np.array(pine_functions.pine_ema(ci, avg))
    wt2 = np.array(pine_functions.sma(wt1, malen))
    wtVwap = wt1 - wt2
    df['wt1'] = wt1
    df['wt2'] = wt2
    df['wtVwap'] = wtVwap


def market_chiper_b(df, config):
    # Only using WT for now
    _wavetrend(df,
               config['wt_ma_source'],
               config['wt_channel_len'],
               config['wt_average_len'],
               config['wt_ma_len'])


def arrayEma(data, n):
    data = np.nan_to_num(data)
    m = (2 / (n + 1))
    f = sum(data[:n]) / n
    emas = [0] * (n - 1)
    emas.append(f)
    for i in range(n, len(data)):
        emas.append((data[i] - emas[i - 1]) * m + (emas[i - 1]))
    return emas


def atr(df, n):
    tr = [df['High'].iloc[0] - df['Low'].iloc[0]]
    for i in range(1, len(df)):
        tr.append(max(df['High'].iloc[i] - df['Low'].iloc[i], abs(df['High'].iloc[i] - df['Close'].iloc[i - 1]),
                      abs(df['Low'].iloc[i] - df['Close'].iloc[i - 1])))
    return arrayEma(tr, n)


def pine_adx(df, adxlen=7, dilen=5):
    up = np.nan_to_num(df['High'] - df['High'].shift(1))
    down = np.nan_to_num(df['Low'].shift(1) - df['Low'])
    plusDM = []
    downDM = []
    for i in range(len(up)):
        plusDM.append(up[i] if up[i] > down[i] and up[i] > 0 else 0)
        downDM.append(down[i] if down[i] > up[i] and down[i] > 0 else 0)
    truerange = np.array(pine_functions.atr(df, dilen))
    plus = np.nan_to_num(pine_functions.pine_ema(np.nan_to_num(plusDM), dilen) / truerange * 100)
    minus = np.nan_to_num(pine_functions.pine_ema(np.nan_to_num(downDM), dilen) / truerange * 100)
    su = np.array(plus) + np.array(minus)
    su[su == 0] = 1
    adx = 100 * np.array(pine_functions.pine_ema(np.nan_to_num(abs(plus - minus)) / su, adxlen))
    df['truerange'] = truerange
    df['plus'] = plus
    df['minus'] = minus
    df['su'] = su
    df['PADX_' + str(adxlen) + '_' + str(dilen)] = adx


def adx(df, adxlen=7, dilen=5):
    up = df['High'] - df['High'].shift(1)
    up.iloc[0] = 0
    down = df['Low'].shift(1) - df['Low']
    down.iloc[0] = 0
    plusDM = []
    downDM = []
    for i in range(len(up)):
        plusDM.append(up.iloc[i] if up.iloc[i] > down.iloc[i] and up.iloc[i] > 0 else 0)
        downDM.append(down.iloc[i] if down.iloc[i] > up.iloc[i] and down.iloc[i] > 0 else 0)
    truerange = np.array(atr(df, dilen))
    plus = np.array(arrayEma(plusDM, dilen)) / truerange * 100
    minus = np.array(arrayEma(downDM, dilen)) / truerange * 100
    su = plus + minus
    su[su == 0] = 1
    adx = 100 * np.array(arrayEma(abs(plus - minus) / su, adxlen))

    df['ADX_' + str(adxlen) + '_' + str(dilen)] = adx


def wave_trend(df, channel_len=10, average_len=10):
    ap = (df['High'] + df['Low'] + df['Close']) / 3
    esa = np.array(pine_functions.pine_ema(ap, channel_len))
    d = np.array(pine_functions.pine_ema(abs(ap - esa), channel_len))
    ci = (ap - esa) / (0.015 * d)
    tci = np.array(pine_functions.pine_ema(ci, average_len))
    df['wave_trend'] = tci


def pD(df, return_array=False):
    haclose = (df['High'] + df['Close'] + df['Low'] + df['Open']) / 4
    haopen = [(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2]
    hahigh = [max(haclose[0], haopen[0], df['High'].iloc[0])]
    halow = [min(haclose[0], haopen[0], df['Low'].iloc[0])]
    for i in range(1, len(df)):
        haopen.append((haopen[i - 1] + haclose[i - 1]) / 2)
        hahigh.append(max(haclose[i], haopen[i], df['High'].iloc[i]))
        halow.append(min(haclose[i], haopen[i], df['Low'].iloc[i]))

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
    if return_array:
        return pD
    df['pD'] = pD


def winkel_reich_god(df, minSlope=1.0001, steepSlope=1.0002, emaLength=48, slopeLookback=10, steepSlopeLookback=10,
                     emaDistanceFilter=1, steepEmaDistanceFilter=1.5):
    haclose = (df['High'] + df['Close'] + df['Low'] + df['Open']) / 4
    haopen = [(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2]
    hahigh = [max(haclose[0], haopen[0], df['High'].iloc[0])]
    halow = [min(haclose[0], haopen[0], df['Low'].iloc[0])]
    for i in range(1, len(df)):
        haopen.append((haopen[i - 1] + haclose[i - 1]) / 2)
        hahigh.append(max(haclose[i], haopen[i], df['High'].iloc[i]))
        halow.append(min(haclose[i], haopen[i], df['Low'].iloc[i]))

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
    pK = np.array(bcwsma(RSV, isig, 1))
    pD = np.array(bcwsma(pK, isig, 1))
    df['pD'] = pD

    # SRSI
    srcChange = pine_functions.change(src)
    up1 = pine_functions.rma(pine_functions.pine_max(srcChange, 0), 9)
    down1 = pine_functions.rma(-1 * np.array(pine_functions.pine_min(srcChange, 0)), 9)
    rsi = []
    for i in range(len(up1)):
        rsi.append(100 if down1[i] == 100 else (0 if up1[i] == 0 else (100 - 100 / (1 + up1[i] / down1[i]))))
    K = pine_functions.sma(pine_functions.stoch(c, hahigh, halow, 7), 5)
    df['stoch_rsi'] = K

    # MACD
    fast_length = 12
    slow_length = 26
    fast_ma = pine_functions.pine_ema(src, fast_length)
    slow_ma = pine_functions.pine_ema(src, slow_length)
    macd = np.array(fast_ma) - np.array(slow_ma)

    df['macd'] = macd

    # GANN Trend
    avghigh = pine_functions.pine_ema(hahigh, 10)
    avglow = pine_functions.pine_ema(halow, 10)
    uptrend = np.array(hahigh) > np.array(avghigh)
    downtrend = np.array(halow) < np.array(avglow)

    df['uptrend'] = uptrend
    df['downtrend'] = downtrend

    ema = np.array(pine_functions.pine_ema(df['Close'], emaLength))
    df['ema'] = ema

    slope = np.array([1])
    for i in range(1, len(ema)):
        slope = np.append(slope, ema[i] / ema[i - 1])
    df['slope'] = np.round(slope, decimals=7)
    sumSlope = np.array(pine_functions.pine_sum(slope, slopeLookback)) / slopeLookback
    steepSumSlope = np.array(pine_functions.pine_sum(slope, steepSlopeLookback)) / steepSlopeLookback
    sumSlope2 = np.array(pine_functions.pine_sum(slope, slopeLookback * 4)) / (slopeLookback * 4)
    df['sumSlope'] = sumSlope
    df['sumSlope2'] = sumSlope2
    slopefilter = [a > minSlope or c > minSlope for a, c in zip(sumSlope, sumSlope2)]
    steepSlopefilter = [a > steepSlope for a in steepSumSlope]
    negSlopeFilter = [.999888 > b for b in sumSlope]
    conSlopeFilter = [a < minSlope and a > .999888 for a in sumSlope]
    emaPercent = np.array(df['Close'] / ema)
    percentFilter = [a < 1 + emaDistanceFilter * .01 for a in emaPercent]
    steepPercentFilter = [a < 1 + steepEmaDistanceFilter * .01 for a in emaPercent]
    closeAboveDMA = [a > b for a, b in zip(df['Close'], ema)]
    df['percentFilter'] = percentFilter
    df['slopefilter'] = slopefilter

    # BUY / SELL
    Buys = [pD and macd and K for pD, macd, K in
            zip(pine_functions.rising(pD, 1), pine_functions.rising(macd, 1), pine_functions.rising(K, 1))]
    Sells = [pD and macd and K for pD, macd, K in
             zip(pine_functions.falling(pD, 1), pine_functions.falling(macd, 1), pine_functions.falling(K, 1))]
    Buy = [b and up for b, up in zip(Buys, uptrend)]
    Sell = [s and dn for s, dn in zip(Sells, downtrend)]
    OscHelper = [0]
    BuySignal = [False]
    SellSignal = [False]
    steepBuySignal = [False]
    negBuySignal = [False]
    conBuySignal = [False]
    actualNegBuys = [False, False]
    actualConBuys = [False, False]
    for i in range(1, len(Buy)):
        OscHelper.append(1 if Buy[i] else (0 if Sell[i] else OscHelper[i - 1]))
        BuySignal.append(Buy[i] and OscHelper[i - 1] == 0 and slopefilter[i] and percentFilter[i])
        SellSignal.append(Sell[i] and OscHelper[i - 1] == 1)
        steepBuySignal.append(Buy[i] and OscHelper[i - 1] == 0 and steepSlopefilter[i] and steepPercentFilter[i])
        negBuySignal.append(
            Buy[i] and OscHelper[i - 1] == 0 and negSlopeFilter[i] and percentFilter[i] and closeAboveDMA[i])
        conBuySignal.append(
            Buy[i] and OscHelper[i - 1] == 0 and conSlopeFilter[i] and percentFilter[i] and closeAboveDMA[i])

    for i in range(2, len(Buy)):
        actualNegBuys.append(negBuySignal[i - 2] and df['Close'].iloc[i - 1] < df['Close'].iloc[i] and (
                    closeAboveDMA[i - 1] or closeAboveDMA[i]) and percentFilter[i])
        actualConBuys.append(conBuySignal[i - 2] and df['Close'].iloc[i - 1] < df['Close'].iloc[i] and (
                    closeAboveDMA[i - 1] or closeAboveDMA[i]) and percentFilter[i])

    df['Buy'] = BuySignal
    df['steepBuySignal'] = steepBuySignal
    df['actualNegBuys'] = actualNegBuys
    df['actualConBuys'] = actualConBuys
    df['negBuySignal'] = negBuySignal
    df['conBuySignal'] = conBuySignal
    df['Sell'] = SellSignal


def tsi(df, r=25, s=13, src='Close'):
    m = (df[src].shift(-1) - df[src]).shift(1).fillna(0)
    tsi = 100 * (np.array(pine_functions.ema(pine_functions.ema(m, r), s)) / np.array(
        pine_functions.ema(pine_functions.ema(abs(m), r), s)))
    df['tsi'] = tsi


def r(df, n=14):
    """Calculate William's %R for the given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    rs = [0] * n
    for i in range(n, len(df)):
        close = df['Close'].loc[i]
        high = df['High'].loc[i - n + 1:i].max()
        low = df['Low'].loc[i - n + 1:i].min()
        rs.append((high - close) / (high - low) * -100)
    df['r'] = rs


def rsi_vol(df, n=14):
    def WiMA(src, n):
        MA_s = [0]
        for i in range(1, len(src)):
            MA_s.append((src[i] + (MA_s[i - 1] * (n - 1)) / n))
        return MA_s

    up = [0]
    do = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            up.append(abs(df['Close'].iloc[i] - df['Close'].iloc[i - 1]) * df['Volume'].iloc[i])
        else:
            up.append(0)
        if df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            do.append(abs(df['Close'].iloc[i] - df['Close'].iloc[i - 1]) * df['Volume'].iloc[i])
        else:
            do.append(0)

    upt = np.array(WiMA(up, n))
    dnt = np.array(WiMA(do, n))
    df['RSI_Vol'] = 100 * (upt / (upt + dnt))


def wt_lb(df, n1=10, n2=21):
    ap = np.array((df['High'] + df['Low'] + df['Close']) / 3)
    esa = np.array(pine_functions.ema(ap, n1))
    d = np.array(pine_functions.ema(np.abs(ap - esa), n1))
    ci = (ap - esa) / (0.015 * d)
    tci = pine_functions.ema(ci, n2)

    wt1 = tci
    wt2 = pine_functions.sma(wt1, 4)


def support_resistance(df):
    srch = np.array(df['High'].values)
    srcl = np.array(df['Low'].values)
    timeframU = 60
    prd = 20
    hc = [srch[0]]
    lc = [srcl[0]]
    for i in range(1, len(srch)):
        start = max(0, i - prd)
        hc.append(srch[i] if np.argmax(srch[start:i + 1]) == len(srch[start:i + 1]) - 1 else hc[i - 1])
        lc.append(srcl[i] if np.argmin(srcl[start:i + 1]) == len(srcl[start:i + 1]) - 1 else lc[i - 1])
    df['hc'] = hc
    df['lc'] = lc


def trix(df, length=18):
    source = np.array(df['Close'])
    df['trix'] = pine_functions.percent_change(
        pine_functions.pine_ema(pine_functions.pine_ema(pine_functions.pine_ema(source, 18), 18), 18)) * 100




"""
Indicators as shown by Peter Bakker at:
https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
"""

"""
25-Mar-2018: Fixed syntax to support the newest version of Pandas. Warnings should no longer appear.
             Fixed some bugs regarding min_periods and NaN.

			 If you find any bugs, please report to github.com/palmbook
"""




def OBV(df, src):
    obv = [0]
    for i in range(len(df) - 1):
        if src[i + 1] - src[i] > 0:
            obv.append(df.loc[i + 1, 'Volume'])
        if src[i + 1] - src[i] == 0:
            obv.append(0)
        if src[i + 1] - src[i] < 0:
            obv.append(-df.loc[i + 1, 'Volume'])
    return obv

def market_god_v6_4(df):
    # HA
    haclose = (df['High'] + df['Close'] + df['Low'] + df['Open']) / 4
    haopen = [(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2]
    hahigh = [max(haclose[0], haopen[0], df['High'].iloc[0])]
    halow = [min(haclose[0], haopen[0], df['Low'].iloc[0])]
    heikUp = [haclose[0] > haopen[0]]
    heikDown = [haclose[0] < haopen[0]]
    for i in range(1, len(df)):
        haopen.append((haopen[i-1] + haclose[i-1]) / 2)
        hahigh.append(max(haclose[i], haopen[i], df['High'].iloc[i]))
        halow.append(min(haclose[i], haopen[i], df['Low'].iloc[i]))
        heikUp.append(haclose[i] > haopen[i])
        heikDown.append(haclose[i] < haopen[i])

    price = haclose
    src = haclose
    c = np.array(haclose)
    obv = OBV(df, haclose)

    #KJD
    ilong = 9
    isig = 3
    def bcwsma(s,l,m):
        _bcwsma = [(m*s[0])/l]
        for i in range(1, len(s)):
            _bcwsma.append( ((m*s[i]) + (l-m)*_bcwsma[i-1]) / l)
        return _bcwsma
    h = np.array(pine_functions.highest(hahigh, ilong))
    l = np.array(pine_functions.lowest(halow, ilong))

    RSV = 100 * ((c-l)/(h-l))
    cl = (c-l)
    hl = (h-l)

    pK = np.array(bcwsma(RSV, isig,1))
    pD = np.array(bcwsma(pK, isig,1))
    pJ = 3 * pK - 2 * pD
    KD = (pK + pD) / 2
    df['RSV'] = RSV
    df['pK'] = pK
    df['l'] = l
    df['h'] = h
    df['c'] = c
    df['hl'] = hl
    df['cl'] = cl
    df['hahigh'] = hahigh
    df['haclose'] = haclose
    df['halow'] = halow
    # highest and lowest
    hgst = np.array(pine_functions.highest(hahigh, 1))
    lwst = np.array(pine_functions.lowest(halow, 1))

    #KBC
    trend = 5
    length = 20
    mult = 2.0
    basis = np.array(pine_functions.sma(src, length))
    dev = np.array(pine_functions.std(src,length))
    ma = np.array(pine_functions.ema(src, length))
    upper = basis + dev
    lower = basis - dev
    useTrueRange = True
    MA5 = np.array(pine_functions.ema(src, length))
    tr = pine_functions.true_range(df)
    rangema = pine_functions.ema(tr, length)
    upperkelt = MA5 + rangema * 1
    lowerkelt = MA5 - rangema * 1
    bbr = (src - lower) / (upper - lower)

    MFlength = 14
    MFsrc = haclose
    changedmf = pine_functions.change(MFsrc)
    lowmf = []
    upmf = []
    for i in range(len(df)):
        lowmf.append(0 if df['Volume'].iloc[i] * changedmf[i] >= 0 else MFsrc[i])
        upmf.append(0 if df['Volume'].iloc[i] * changedmf[i] <= 0 else MFsrc[i])
    upperMF = pine_functions.pine_sum(upmf, MFlength)
    lowerMF = pine_functions.pine_sum(lowmf, MFlength)

    #mf = pine_functions.rsi(upperMF, lowerMF)
    # rsi for two sequence is not explained on pine's documentation
    # MRUp = pine_functions.crossover(mf,0)
    # MRDwn = pine_functions.crossunder(mf,0)

    # PARABOLIC SAR
    out2 = pine_functions.sar(df)

    # SRSI
    srcChange = pine_functions.change(src)
    up1 = pine_functions.rma(pine_functions.pine_max(srcChange, 0), 9)
    down1 = pine_functions.rma(-1 * np.array(pine_functions.pine_min(srcChange, 0)), 9)
    rsi = []
    for i in range(len(up1)):
        rsi.append(100 if down1[i] == 100 else (0 if up1[i] == 0 else (100 - 100 / (1 + up1[i]/down1[i]))))
    K = pine_functions.sma(pine_functions.stoch(c, hahigh, halow, 7), 5)
    D = pine_functions.sma(K, 3)
    smoothK = 3
    smoothD = 3
    lengthRSI = 14
    lengthStoch = 14
    rsi1 = pine_functions.rsi(src, lengthRSI)
    k = pine_functions.ema(pine_functions.stoch(rsi1,rsi1,rsi1, lengthStoch),smoothK)
    d = pine_functions.rsi(k, smoothD)

    # GUPPY
    len1 = 3
    len2 = 5
    len3 = 8
    len4 = 10
    len5 = 12
    len6 = 15

    # SLOW EMA
    len7 = 30
    len8 = 35
    len9 = 40
    len10 = 45
    len11 = 50
    len12 = 60

    # FAST EMA
    ema1 = pine_functions.ema(src, len1)
    ema2 = pine_functions.ema(src, len2)
    ema3 = pine_functions.ema(src, len3)
    ema4 = pine_functions.ema(src, len4)
    ema5 = pine_functions.ema(src, len5)
    ema6 = pine_functions.ema(src, len6)
    # SLOW EMA
    ema7 = pine_functions.ema(src, len7)
    ema8 = pine_functions.ema(src, len8)
    ema9 = pine_functions.ema(src, len9)
    ema10 = pine_functions.ema(src, len10)
    ema11 = pine_functions.ema(src, len11)
    ema12 = pine_functions.ema(src, len12)
    tema = 2 * (np.array(ema1) - np.array(ema2)) + np.array(ema3)

    # FAST EMA COLOR RULES
    GupUp = [a > b and c for a,b,c in zip(ema1, ema2, pine_functions.rising(obv,1))]
    GupDwn = [a < b and c for a,b,c in zip(ema1, ema2, pine_functions.falling(obv,1))]
    # SLOW EMA COLOR RULES
    GuppUp = [a > b and b > c for a,b,c in zip(ema7, ema8, ema9)]
    GuppDwn = [a < b and b < c for a,b,c in zip(ema7, ema8, ema9)]
    RSI6 = pine_functions.rsi(haclose,6)
    RSI12 = pine_functions.rsi(haclose,12)
    RSI24 = pine_functions.rsi(haclose,20)
    RSIUp = pine_functions.crossover(RSI12,RSI24)
    RSIDwn = pine_functions.crossunder(RSI12,RSI24)
    KDJUp = pine_functions.crossover(pJ,pD)
    KDJDwn = pine_functions.crossunder(pJ,pD)
    # PIVOTS
    #TODO ADD PIVOTS
    n1 = 12
    n2ma = 2 * np.array(pine_functions.wma(haclose, int(round(n1 / 2))))
    nma= pine_functions.wma(haclose,n1)
    diff= np.array(n2ma) - np.array(nma)
    sqn = int(round(math.sqrt(n1)))
    C=5
    n2ma6= 2 * np.array(pine_functions.wma(haopen, int(round(C/2))))
    nma6= np.array(pine_functions.wma(haopen,C))
    diff6=n2ma6-nma6
    sqn6= int(round(math.sqrt(C)))
    a1= pine_functions.wma(diff6,sqn6)
    a= pine_functions.wma(diff,sqn)
    a1greata = np.array([a > b for a,b in zip(a1, a)]).astype(int)
    a1lessa = np.array([a < b for a,b in zip(a1, a)]).astype(int)
    gains= np.array(pine_functions.pine_sum(a1greata, 1))
    losses = np.array(pine_functions.pine_sum(a1lessa, 1))
    cmo = 100 * (gains - losses) / (gains + losses)
    H = pine_functions.highest(df['High'].values,2)
    hdev = pine_functions.dev(H,len5)
    hpivot = [H[0]]
    for i in range(1, len(H)):
        hpivot.append(hpivot[i-1] if hdev[i] > 0 else H[i])
    L = pine_functions.highest(df['Low'].values,2)
    ldev = pine_functions.dev(L,2)
    lpivot = [L[0]]
    for i in range(1, len(H)):
        lpivot.append(lpivot[i-1] if ldev[i] > 0 else L[i])


    # MACD
    fast_length = 12
    slow_length = 26
    signal_length = 9
    fast_ma = pine_functions.pine_ema(src, fast_length)
    slow_ma = pine_functions.pine_ema(src, slow_length)
    macd = np.array(fast_ma) - np.array(slow_ma)
    signal = pine_functions.pine_ema(macd, signal_length)
    hist = np.array(macd) - np.array(signal)
    MACUp = pine_functions.crossover(hist, 0)
    MACDown = pine_functions.crossunder(hist, 0)
    '''
    # SUPER TREND
    Factor=3
    ATR=7
    RSI = 7
    # SUPER TREND ATR
    Upp= np.array(hl2) - (Factor * np.array(pine_functions.atr(df, ATR)))
    Dnn= np.array(hl2) + (Factor * np.array(pine_functions.atr(df, ATR)))
    TUp = [Upp[0]]
    TDown = [Dnn[0]]
    for i in range(1, len(Upp)):
        TUp.append(max(Upp[i],TUp[i-1]) if haclose[i-1]>TUp[i-1] else Upp[i])
        TDown.append(min(Dnn,TDown[i-1]) if haclose[i-1]<TDown[i-1] else Dnn[i])

    # TODO finish this translation

    Trendd = haclose > TDown[1] ? 1: haclose< TUp[1]? -1: nz(Trendd[1],1)
    Tsll = Trendd==1? TUp: TDown
    ep = 2 * RSI - 1
    auc = pine_functions.ema( max( src - src[1], 0 ), ep )
    adc = pine_functions.ema( max( src[1] - src, 0 ), ep )
    X1 = (RSI - 1) * ( adc * 70 / (100-70) - auc)
    ub = iff( X1 >= 0, src + X1, src + X1 * (100-70)/70 )
    X2 = (RSI - 1) * ( adc * 30 / (100-30) - auc)
    lb = iff( X2 >= 0, src + X2, src + X2 * (100-30)/30 )
    '''
    STUp= pine_functions.crossover(haclose,lower)
    STDwn = pine_functions.crossunder(haclose,upper)


    # Hi-Lo
    hGst = pine_functions.highest(hahigh,1)
    lWst = pine_functions.lowest(halow,1)

    # GANN Trend
    avghigh = pine_functions.pine_ema(hahigh, 10)
    avglow = pine_functions.pine_ema(halow, 10)
    uptrend = np.array(hahigh) > np.array(avghigh)
    downtrend = np.array(halow) < np.array(avglow)

    df['avglow'] = avglow
    df['avghigh'] = avghigh

    # DEMA TEMA
    e1 = pine_functions.ema(haclose,9)
    e2 = pine_functions.ema(e1,9)
    ema101 = pine_functions.ema(haclose,9)
    ema202 = pine_functions.ema(ema101,9)
    ema3303 = pine_functions.ema(ema202,9)
    tema2 = 3 * (np.array(ema101) - np.array(ema202)) + np.array(ema3303)
    dema = 2*np.array(e1) - np.array(e2)

    # BUY / SELL
    Buys = [pD and macd and K for pD,macd,K in zip(pine_functions.rising(pD,1), pine_functions.rising(macd,1), pine_functions.rising(K,1))]
    Sells = [pD and macd and K for pD,macd,K in zip(pine_functions.falling(pD,1), pine_functions.falling(macd,1), pine_functions.falling(K,1))]
    Buy = [b and up for b, up in zip(Buys, uptrend)]
    Sell = [s and dn for s, dn in zip(Sells, downtrend)]
    OscHelper = [0]
    BuySignal = [False]
    SellSignal = [False]
    for i in range(1, len(Buy)):
        OscHelper.append(1 if Buy[i] else (0 if Sell[i] else OscHelper[i-1]))
        BuySignal.append(Buy[i] and OscHelper[i-1] == 0)
        SellSignal.append(Sell[i] and OscHelper[i-1] == 1)


    df['Sells'] = Sells
    df['Sell1'] = Sell
    df['Buy'] = BuySignal
    df['Sell'] = SellSignal
    df['pD'] = pD
    df['macd'] = macd
    df['K'] = K
    df['uptrend'] = uptrend
    df['downtrend'] = downtrend




def std(x, n):
    std = []
    for i in range(1, n):
        std.append(np.std(x[0:i]))
    for i in range(n, len(x) + 1):
        std.append(np.std(x[i-n:i]))
    return std

def true_range(df):
    tr = [0]
    for i in range(1, len(df)):
        tr.append(max(df['High'].iloc[i] - df['Low'].iloc[i],
                  abs(df['High'].iloc[i] - df['Close'].iloc[i-1]),
                  abs(df['Low'].iloc[i] - df['Close'].iloc[i-1])))
    return tr

def vapi(df, length=10):
    x = (2 * df['Close'] - df['High'] - df['Low']) /  (df['High'] - df['Low'])
    tva = (df['Volume']*x).rolling(length).sum()
    tv = (df['Volume']).rolling(length).sum()
    va = 100 * tva / tv
    df['vapi'] = va

def detectWhipsaw(df, start, n):
    change = max(df['Close'].iloc[start - n:start]) / min(df['Close'].iloc[start - n:start])
    whipStart = -1
    whipEnd = -1
    if  change > 1.1:
        indices = detect_peaks(df['Close'].iloc[start - n:start], mpd=5, valley=False)
        for i in indices:
            if df['Close'].iloc[i + start - n] == max(df['Close'].iloc[start - n:start]):
                whipEnd = i + start - n
            elif df['Close'].iloc[i+ start - n] == min(df['Close'].iloc[start - n:start]):
                whipStart = i + start - n
        print(indices + start - n)
        print('Large Spike')
    elif change < 0.9:
        print('Large Drop')
    return (whipStart, whipEnd)

def ATR_Trailing_Stop_Loss(df, nATRPeriod=5, nATRMultip=3.5):
    xATR = np.array(arrayEma(true_range(df), nATRPeriod))
    nLoss =  xATR * nATRMultip
    xATRTrailingStop = [0]
    for i in range(1, len(xATR)):
        if df['Close'].iloc[i] > xATRTrailingStop[i-1] and df['Close'].iloc[i-1] > xATRTrailingStop[i-1]:
            xATRTrailingStop.append(max(xATRTrailingStop[i-1], df['Close'].iloc[i] - nLoss[i]));
        elif df['Close'].iloc[i] < xATRTrailingStop[i-1] and df['Close'].iloc[i-1] < xATRTrailingStop[i-1]:
            xATRTrailingStop.append(min(xATRTrailingStop[i-1], df['Close'].iloc[i] + nLoss[i]));
        elif df['Close'].iloc[i] > xATRTrailingStop[i-1]:
            xATRTrailingStop.append(df['Close'].iloc[i] - nLoss[i])
        else:
            xATRTrailingStop.append(df['Close'].iloc[i] + nLoss[i])
    df['ATR_SL'] = xATRTrailingStop

def ppo_divergence(df, indicator, long_term_div=True, div_lookback_period=55, fastLength=12, signalLength=9, smoother=2):
    bullishPrice = df['Low']
    d = arraySma(df[indicator], smoother)

    priceMin = ((bullishPrice.shift(1) < bullishPrice and bullishPrice.shift(1) < bullishPrice.shift(2))
    or (bullishPrice.shift(1) == bullishPrice.shift(2) and bullishPrice.shift(1) < bullishPrice and bullishPrice.shift(3) < bullishPrice)
    or (bullishPrice.shift(1) == bullishPrice.shift(2) and bullishPrice.shift(1) == bullishPrice.shift(3) and bullishPrice.shift(1) < bullishPrice and bullishPrice.shift(1) < bullishPrice.shift(4))
    or (bullishPrice.shift(1) == bullishPrice.shift(2) and bullishPrice.shift(1) == bullishPrice.shift(3) and bullishPrice.shift(1) == bullishPrice.shift(4) and bullishPrice.shift(1) < bullishPrice and bullishPrice.shift(1) < bullishPrice.shift(5)))

    oscMins = [False, False]
    for i in range(len(d)):
        oscMins.append(d[i] > d[i-1] and d[i-1] < d[i-2])

    bearishPrice = df['High']
    priceMax = ((bearishPrice.shift(1) > bearishPrice and bearishPrice.shift(1) > bearishPrice.shift(2))
    or (bearishPrice.shift(1) == bearishPrice.shift(2) and bearishPrice.shift(1) > bearishPrice and bearishPrice.shift(3) > bearishPrice)
    or (bearishPrice.shift(1) == bearishPrice.shift(2) and bearishPrice.shift(1) == bearishPrice.shift(3) and bearishPrice.shift(1) > bearishPrice and bearishPrice.shift(1) > bearishPrice.shift(4))
    or (bearishPrice.shift(1) == bearishPrice.shift(2) and bearishPrice.shift(1) == bearishPrice.shift(3) and bearishPrice.shift(1) == bearishPrice.shift(4) and bearishPrice.shift(1) > bearishPrice and bearishPrice.shift(1) > bearishPrice.shift(5)))

    oscMaxes = [False, False]
    for i in range(len(d)):
        oscMaxes.append(d[i] < d[i-1] and d[i-1] > d[i-2])

def volumeSpike(df):
    df['tradesEMA'] = arrayEma(df['Number of trades'].values,8)
    df['volumeEMA'] = arrayEma(df['Volume'].values,8)

def dmi(df):
    def wwma(l, p):
        ma = [0]
        for i in range(1, len(p)):
            ma.append((ma[i-1] * (l - 1) + p[i]) / l)
        return np.array(ma)
    DMIlength = 10
    Stolength = 10
    os = 10
    ob = 90

    hiDiff = [0]
    loDiff = [0]
    plusDM = [0]
    minusDM = [0]
    for i in range(1, len(df)):
        hiDiff.append(df['High'].iloc[i] - df['High'].iloc[i-1])
        loDiff.append(df['Low'].iloc[i] - df['Low'].iloc[i-1])
        if (hiDiff[i] > loDiff[i]) and (hiDiff[i] > 0):
            plusDM.append(hiDiff[i])
        else:
            plusDM.append(0)
        if (loDiff[i] > hiDiff[i]) and (loDiff[i] > 0):
            minusDM.append(loDiff[i])
        else:
            minusDM.append(0)

    ATR = wwma(DMIlength, true_range(df))
    PlusDI = 100 * wwma(DMIlength,plusDM) / ATR
    MinusDI = 100 * wwma(DMIlength,minusDM) / ATR
    osc = PlusDI - MinusDI

    PlusDI[0] = 0
    MinusDI[0] = 0
    osc[0] = 0

    ll = []
    hh = []
    for i in range(1, Stolength):
        ll.append(min(osc[0:i]))
        hh.append(max(osc[0:i]))
    for i in range(Stolength, len(df) + 1):
        ll.append(min(osc[i-Stolength:i]))
        hh.append(max(osc[i-Stolength:i]))
    ll = np.array(ll)
    hh = np.array(hh)
    Stoch = [0] * Stolength
    for i in range(Stolength, len(osc)):
        Stoch.append(sum(osc[i-Stolength:i]-ll[i-Stolength:i]) / sum((hh[i-Stolength:i]-ll[i-Stolength:i])) * 100)

    df['dmi'] = Stoch

def moving_average(df, n):
    """Calculate the moving average for the given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    df = df.join(MA)
    return df

def tfi(df, n=20, coef=0.1, vcoef=2.5,signalLength=20):
    typical = (df['Close'] + df['Low'] + df['High']) / 3
    mf = [0]
    inter = [0]
    for i in range(1, len(typical)):
        inter.append(math.log(typical[i]) - math.log(typical[i-1]))
        mf.append(typical[i] - typical[i-1])

    vinter = std(np.array(inter), 30)
    cutoff = df['Close'] * vinter * coef
    vave = np.roll(arraySma( df['Volume'], n ),1)
    vave[0] = 0
    vmax = vave * vcoef
    vc = []
    for i in range(len(vmax)):
        if df['Volume'].iloc[i] < vmax[i]:
            vc.append(df['Volume'].iloc[i])
        else:
            vc.append(vmax[i])

    vcp = []
    for i in range(len(mf)):
        if mf[i] > cutoff[i]:
            vcp.append(vc[i])
        else:
            if mf[i] < -cutoff[i]:
                vcp.append(-vc[i])
            else:
                vcp.append(0)

    vfiSum = [0] * (n - 1)
    for i in range(n,len(vcp)+1):
        vfiSum.append(sum(vcp[i-n:i]))
    vfi = vfiSum / vave
    df['vfi'] = vfi
    df['vfima'] = arrayEma(vfi, signalLength)

def cloud(df, conversionPeriods=9, basePeriods=26, lagginSpan2Periods=52,displacement=26):
    def donchian(df, period):
        don = []
        for i in range(1,period):
            don.append((min(df['Low'].iloc[0:i]) + max(df['High'].iloc[0:i])) / 2)
        for i in range(period, len(df) + 1):
            don.append((min(df['Low'].iloc[i-period:i]) + max(df['High'].iloc[i-period:i])) / 2)
        return don

    conversionLine = donchian(df, conversionPeriods)
    baseline = donchian(df, basePeriods)
    leadline1 = (np.array(conversionLine) + np.array(baseline)) / 2
    leadline2 = donchian(df, lagginSpan2Periods)

    df['conversionline'] = conversionLine
    df['baseline'] = baseline
    df['leadline1'] = leadline1
    df['leadline2'] = leadline2

def vort(df, n=14):
    vmp = [0] * n
    vmm = [0] * n
    for i in range(n,len(df)):
        vmp.append(sum( abs( np.array(df['High'].iloc[i-n:i]) - np.array(df['Low'].iloc[i-n+1:i+1] )) ))
        vmm.append(sum( abs( np.array(df['Low'].iloc[i-n:i]) - np.array(df['High'].iloc[i-n+1:i+1] )) ))
    st = average_true_range(df,1)['ATR_1']
    for i in range(n, len(st)):
        st[i] = sum(st[i-n:i])
    vip = vmp
    vim = vmm
    df['vip'] = vip
    df['vim'] = vim

def wtc(df, n=21, c=10):
    ap = np.nan_to_num((df['Close'] + df['Low'] + df['High']) / 3) * 1000
    esa = np.nan_to_num(arrayEma(ap, c))
    d = np.nan_to_num(arrayEma((ap - esa), c))
    ci = np.nan_to_num(np.array(ap - esa) / (np.array(d) * 0.015))
    tci = arrayEma(ci, n)
    wti = arraySma(tci, 4)
    df['wtc_tci'] = tci
    df['wtc_wti'] = wti

def stoch_mtm(df, a=10, b=3):
    ll = []
    hh = []
    for i in range(1, a):
        ll.append(min(df['Low'].iloc[0:i]))
        hh.append(max(df['High'].iloc[0:i]))
    for i in range(a, len(df) + 1):
        ll.append(min(df['Low'].iloc[i-a:i]))
        hh.append(max(df['High'].iloc[i-a:i]))
    ll = np.array(ll)
    hh = np.array(hh)
    diff = hh - ll
    rdiff = df['Close'] - (hh+ll) / 2
    avgrel = arrayEma(arrayEma(rdiff,b),b)
    avgdiff = arrayEma(arrayEma(diff,b),b)
    SMI = []
    for i in range(len(avgdiff)):
        SMI.append((avgrel[i]/(avgdiff[i]/2)*100) if avgdiff[i] != 0 else 0)
    SMIsignal = arrayEma(SMI,b)
    emasignal = arrayEma(SMI, 10)
    df['stoch_mtm_SMI'] = SMIsignal
    df['stoch_mtm_EMA'] = emasignal

def rmi(data):
    le = 20
    mom = 5
    up = [0] * (mom - 1)
    dn = [0] * (mom - 1)
    rmi = [0] * (mom -1)
    for i in range(mom-1, len(data)):
        up.append(max(data['Close'].iloc[i] - data['Close'].iloc[i-mom],0))
        dn.append(max(data['Close'].iloc[i-mom] - data['Close'].iloc[i],0))

    up = arrayEma(up,le)
    dn = arrayEma(dn,le)

    for i in range(mom-1, len(data)):
        if dn[i] == 0:
            rmi.append(0)
        else:
            rmi.append(100 - 100 / (1 + up[i] / dn[i]))
    data['rmi'] = rmi

def FibBands(data, n=60, mult=3):
    hcl3 = (data['Close'] + data['Low'] + data['High']) / 3
    basis = np.array(arrayVwma(data['Close'], data['Volume'], n))
    std = []
    for i in range(1, n):
        std.append(np.std(hcl3[0:i]))
    for i in range(n, len(hcl3) + 1):
        std.append(np.std(hcl3[i-n:i]))
    dev = np.array(std) * mult

    data['fib_dev'] = dev
    upper_1= basis + (0.236*dev)
    upper_2= basis + (0.382*dev)
    upper_3= basis + (0.5*dev)
    upper_4= basis + (0.618*dev)
    upper_5= basis + (0.764*dev)
    upper_6= basis + (1*dev)
    lower_1= basis - (0.236*dev)
    lower_2= basis - (0.382*dev)
    lower_3= basis - (0.5*dev)
    lower_4= basis - (0.618*dev)
    lower_5= basis - (0.764*dev)
    lower_6= basis - (1*dev)

    data['upper_1'] = upper_1
    data['upper_2'] = upper_2
    data['upper_3'] = upper_3
    data['upper_4'] = upper_4
    data['upper_5'] = upper_5
    data['upper_6'] = upper_6
    data['basis'] = basis
    data['lower_1'] = lower_1
    data['lower_2'] = lower_2
    data['lower_3'] = lower_3
    data['lower_4'] = lower_4
    data['lower_5'] = lower_5
    data['lower_6'] = lower_6

def calculate_channel(df, n=60, startFromEnd=80, squeeze=1, ext=20):
    yHat = df['Close'].iloc[len(df) - startFromEnd: len(df) - startFromEnd + n].mean()
    xHat = np.mean(np.array(range(n)))

    x = np.array(range(n)) - xHat
    y = np.array(df['Close'].iloc[len(df) - startFromEnd: len(df) - startFromEnd + n].values) - yHat
    x2 = sum(x*x)
    xy = sum(x*y)

    m = xy / x2

    b = yHat - m*xHat

    extend = np.array(range(n+ext))
    mid = extend*m + b
    maxDist = max(abs(np.array(df['Close'].iloc[len(df) - startFromEnd: len(df) - startFromEnd + n].values) - (np.array(range(n))*m + b))) * squeeze
    top = mid+maxDist
    bot = mid-maxDist
    return top, mid, bot

def regression(df, n=60, column='Close'):
    slopes = [0]*n
    for i in range(n, len(df)):
        slopes.append(calculate_regression(df, i-n, i))

    df[column+'_regression_' + str(n)] = np.array(slopes)

def calculate_regression(df, start, end, column='Close'):
    yHat = df[column].iloc[start:end].mean()
    xHat = np.mean(np.array(range(start, end)))

    x = np.array(range(start, end)) - xHat
    y = np.array(df[column].iloc[start:end].values) - yHat
    x2 = sum(x*x)
    xy = sum(x*y)

    return xy / x2

# 5 minutes min distance between peaks
# 10 minutes lingering divergence
# current indicator value has to be higher than first price peak spot to be valid
# wicks, +.3% or lower to be valid
# 30 minutes before first buy
def threshold_divergence(df, thresh=1.003, indicator='tdi_fastMA_21', indThresh=50, mindist=15, maxdist=60, valid=10):
    div = [False] * (maxdist + valid)
    for i in range(len(div), len(df)):
        if df[indicator].iloc[i] < indThresh:
            div.append(threshold_divergence_detection(df, thresh, i, indicator, mindist=5, maxdist=30, valid=10))
        else:
            div.append(False)
    df['threshold_divergence_' + indicator] = div

def threshold_divergence_detection(df, thresh, i, indicator, mindist=5, maxdist=30, valid=10):
    inds = argrelextrema(df['Close'].iloc[i-valid:i].values, np.less) [0]
    if len(inds) > 0:
        end = i - valid + inds[len(inds) - 1]
        secLick = df['Close'].iloc[end]
        inds = argrelextrema(df['Close'].iloc[end-maxdist-mindist:end-mindist].values, np.less)[0]
        if len(inds) > 0:
            start = end - maxdist + inds[len(inds) - 1]
            firstLick = df['Close'].iloc[start]
            if firstLick*thresh > secLick:
                return df[indicator].iloc[start] < df[indicator].iloc[end]
    return False

def divergence(df, n=45, column='Close'):
    div = [False] * n
    for i in range(n, len(df)):
        div.append(divergence_detection(df, i-n, i, column))
    df[column+'_divergence'] = div

def divergence_detection(df, start, end, column):
    values = list(df[column].iloc[start:end].values)
    lowInd = values.index(min(values))
    low = min(values)


    firstHalf = start + lowInd - 20
    firstVal = low
    if firstHalf > start:
        firstVal = min(np.array(df[column].iloc[start:firstHalf].values))
    secHalf = start + lowInd + 20
    secVal = low
    if secHalf < end:
        secVal = min(np.array(df[column].iloc[secHalf:end].values))
    return firstVal < low or low < secVal


def tdi(df, rsiPeriod=21,bandLength=34,lengthrsipl=7,lengthtradesl=2):
    r = relative_strength_index(df, n=rsiPeriod, ret=True)

    ma = arraySma(r,bandLength)

    std = [0] * (bandLength)
    for i in range(bandLength, len(r)):
        std.append(np.std(r[i-bandLength:i]))

    std = np.array(std)

    offs =  std * 1.6185
    up = ma + offs
    dn = ma - offs
    mid = (up + dn) / 2
    fastMA = arraySma(r, lengthrsipl)
    slowMA = arraySma(r, lengthtradesl)
    df['tdi_up'] = up
    df['tdi_dn'] = dn
    df['tdi_mid'] = mid
    df['tdi_fastMA_' + str(rsiPeriod)] = fastMA
    df['tdi_slowMA'] = slowMA


def williams_r(df, n):
    """Calculate William's %R for the given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    rs = [0] * n
    for i in range(n, len(df)):
        close = df['Close'].loc[i]
        high = df['High'].loc[i-n:i].max()
        low = df['Low'].loc[i-n:i].min()
        rs.append((high - close) / (high - low) * -100)
    r = pd.Series(rs,name='R')
    df = df.join(r)
    return df

def ttf(df, n=15):
    high = []
    low = []
    ttf = [0] * 15
    for i in range(0, n):
        high.append(df['High'].loc[0:i].max())
        low.append(df['Low'].loc[0:i].min())
    for i in range(n, len(df)):
        high.append(df['High'].loc[i-n:i].max())
        low.append(df['Low'].loc[i-n:i].min())
        bp = high[i] - low[i-n]
        sp = high[i-n] - low[i]

        ttf.append(100 * (bp - sp) / ( 0.5*( bp + sp) ))
    df['ttf'] = ttf

def exponential_moving_average(df, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    df = df.join(EMA)
    return df

def sma(df,n, column='Close', name='sma', ret=False):
    if ret:
        return df[column].rolling(n).mean()
    else:
        df[name+str(n)] = df[column].rolling(n).mean()

def wma(df,n, column='Close', name='wma', ret=False):
    wmas = [0] * (n - 1)
    for i in range(n, len(df)):
        s = 0
        w = 0
        for j in range(1,n + 1):
            s = s + (df[column].iloc[i - n + j - 1] * j)
            w = w + j
        wmas.append(s/w)
    if ret:
        return wmas
    else:
        df[name + str(n)] = wmas

def ema(df,n, column='Close', name='ema', ret=False):
    m = (2 / (n + 1))
    f = df[column].iloc[0:n].sum() / n
    emas = [0] * (n - 1)
    emas.append(f)
    for i in range(n, len(df)):
        emas.append((df[column].iloc[i] -emas[i - 1]) * m + (emas[i - 1]))
    if ret:
        return emas
    else:
        df[name + str(n)] = emas

def arrayEma(data,n):
    data = np.nan_to_num(data)
    m = (2 / (n + 1))
    f = sum(data[:n]) / n
    emas = [0] * (n - 1)
    emas.append(f)
    for i in range(n, len(data)):
        emas.append((data[i] -emas[i - 1]) * m + (emas[i - 1]))
    return emas
def arraySma(data,n):
    data = np.nan_to_num(data)
    smas = [0] * (n-1)
    for i in range(n, len(data) + 1):
        smas.append(sum(data[i-n:i]) /n)
    return smas

def arrayVwma(data, volume, n):
    vwma = [0] * (n-1)
    for i in range(n, len(data) + 1):
        vwma.append( (sum(data[i-n:i] * volume[i-n:i]) / n) / (sum(volume[i-n:i])/n))
    return vwma

def true_strength_index(df, src='Close', r=25, s=13):
    m = (df[src].shift(-1) - df[src]).shift(1).fillna(0)
    tsi=100*(np.array(arrayEma(arrayEma(m,r),s))/np.array(arrayEma(arrayEma(abs(m), r),s)))
    df['tsi'] = tsi

def derivative_oscillator(df, rLen=14, ema1=5, ema2=3, p=9):
    s1=arrayEma(arrayEma(relative_strength_index(df, rLen, ret=True), ema1),ema2)
    s2=np.array(s1) - np.array(arraySma(s1,p))
    df['DO'] = s2

def rsi_vol(df, n=14, return_array=False):
    def WiMA(src, n):
        MA_s = [0]
        for i in range(1, len(src)):
            MA_s.append((src[i] + (MA_s[i-1] * (n-1)) / n))
        return MA_s

    up = [0]
    do = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i -1]:
            up.append(abs(df['Close'].iloc[i] - df['Close'].iloc[i -1]) * df['Volume'].iloc[i])
        else:
            up.append(0)
        if df['Close'].iloc[i] < df['Close'].iloc[i -1]:
            do.append(abs(df['Close'].iloc[i] - df['Close'].iloc[i -1]) * df['Volume'].iloc[i])
        else:
            do.append(0)

    upt= np.array(WiMA(up,n))
    dnt= np.array(WiMA(do,n))

    df['RSI_Vol'] = 100*(upt/(upt+dnt))

    if return_array:
        return df['RSI_Vol']

def atr(df,n):
    tr = [df['High'].iloc[0]-df['Low'].iloc[0]]
    for i in range(1, len(df)):
        tr.append(max(df['High'].iloc[i]-df['Low'].iloc[i], abs(df['High'].iloc[i]-df['Close'].iloc[i-1]),abs(df['Low'].iloc[i]-df['Close'].iloc[i-1])))
    return arrayEma(tr, n)

def cvi(df,n=3):
    vc = arraySma((df['High'] - df['Low']) / 2,n)
    df['cvi'] = (df['Close'] - vc) / atr(df,n)

'''
wave trend oscillator krypt
'''
def wvt():
    PI=3.14159265359
    def dropn(src, n):
        return np.nan if np.isnan(src[n]) else src

    def EhlersSuperSmoother(src, lower):
        a1 = math.exp(-PI * math.sqrt(2) / lower)
        coeff2 = 2 * a1 * math.cos(math.sqrt(2) * PI / lower)
        coeff3 = -pow(a1, 2)
        coeff1 = (1 - coeff2 - coeff3) / 2
        filt = [0,0]
        for i in range(3, len(src)):
            filt.append(coeff1 * src[i] + src[i - 1] + coeff2 * filt[i-1] + coeff3 *filt[i-2])
        return filt

    def xema(src, n):
        m = (2 / (n + 1))
        f = src[0:n].sum() / n
        emas = [0] * (n - 1)
        emas.append(f)
        for i in range(n, len(src)):
            emas.append((src[i] -emas[i - 1]) * m + (emas[i - 1]))
        return emas

    def EhlersEmaSmoother(sig, smoothK, smoothP):
        EhlersSuperSmoother(xema(sig, smoothK), smoothP)

    def step(xs, vals):
        res = [vals[0] if xs[0] else 0]
        for i in range(1, len(vals)):
            res.append(vals[i] if xs[i] else res[i - 1])
        return res

def momentum(df, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    M = pd.Series(df['Close'].diff(n), name='Momentum_' + str(n))
    df = df.join(M)
    return df


def rate_of_change(df, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    df['ROC_' + str(n)] = pd.Series(M / N)
    return df


def average_true_range(df, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'High'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'Low'], df.loc[i, 'Close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean(), name='ATR_' + str(n))
    df = df.join(ATR)
    return df


def bollinger_bands(df, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df['Close'].rolling(n, min_periods=n).std())

    df['B_Mid'] = MA
    df['B_Top'] = MA + MSD*2.25
    df['B_Bot'] = MA - MSD*2.25
    '''
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)
    return df
    '''

def parabolic_sar(df, mx=0.2, acc=0.02):
    columns = ['EP', 'Acc', 'Initial PSAR','PSAR', 'PSAR-EP*ACC', 'Trend']
    psarDF = pd.DataFrame(columns=columns)

    psarDF = psarDF.append(pd.Series(pd.Series([df['Low'].iloc[0], acc, df['High'].iloc[0], (df['High'].iloc[0] - df['Low'].iloc[0]) * acc, (df['High'].iloc[0] - df['Low'].iloc[0]) * acc,'Falling'], index=columns )), ignore_index=True)
    for i in range(1, len(df)):
        initial = 0
        psar = 0
        ep = 0
        newAcc = acc

        if psarDF['Trend'].iloc[i-1] == 'Falling':
            initial = max(psarDF['PSAR'].iloc[i-1] - psarDF['PSAR-EP*ACC'].iloc[i-1], df['High'].iloc[i-1], df['High'].iloc[0 if i-2 < 0 else i-2])
        else:
            initial = min(psarDF['PSAR'].iloc[i-1] - psarDF['PSAR-EP*ACC'].iloc[i-1], df['Low'].iloc[i-1], df['Low'].iloc[0 if i-2 < 0 else i-2])

        if psarDF['Trend'].iloc[i-1] == 'Falling' and df['High'].iloc[i] < initial:
            psar = initial
        elif psarDF['Trend'].iloc[i-1] == 'Rising' and df['Low'].iloc[i] > initial:
            psar = initial
        elif psarDF['Trend'].iloc[i-1] == 'Falling' and df['High'].iloc[i] >= initial:
            psar = psarDF['EP'].loc[i-1]
        elif psarDF['Trend'].iloc[i-1] == 'Rising' and df['Low'].iloc[i] <= initial:
            psar = psarDF['EP'].iloc[i-1]

        trend = 'Falling' if psar > df['Close'].iloc[i] else 'Rising'

        if trend == 'Falling':
            ep = min(psarDF['EP'].iloc[i-1], df['Low'].iloc[i])
        elif trend == 'Rising':
            ep = max(psarDF['EP'].iloc[i-1], df['High'].iloc[i])

        if trend == psarDF['Trend'].iloc[i-1]:
            if ep != psarDF['EP'].iloc[i-1]:
                newAcc = psarDF['Acc'].iloc[i-1]
            else:
                newAcc = min(acc + psarDF['Acc'].iloc[i-1], mx)
        psarEPAcc = (psar - ep) *newAcc

        psarDF = psarDF.append(pd.Series(pd.Series([ep, newAcc, initial, psar, psarEPAcc,trend], index=columns )), ignore_index=True)

    df['PSAR'] = psarDF['PSAR']


def ppsr(df):
    """Calculate Pivot Points, Supports and Resistances for given data

    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
    R1 = pd.Series(2 * PP - df['Low'])
    S1 = pd.Series(2 * PP - df['High'])
    R2 = pd.Series(PP + df['High'] - df['Low'])
    S2 = pd.Series(PP - df['High'] + df['Low'])
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df


def stochastic_oscillator_k(df):
    """Calculate stochastic oscillator %K for given data.

    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    df = df.join(SOk)
    return df


def stochastic_oscillator_d(df, n):
    """Calculate stochastic oscillator %D for given data.
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SO%d_' + str(n))
    df = df.join(SOd)
    return df


def trix(df, n):
    """Calculate TRIX for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    ema(df,n, name='Trix')
    ema(df,n, column='Trix' + str(n), name='Trix')
    ema(df,n, column='Trix' + str(n), name='Trix')


def average_directional_movement_index(df, n, n_ADX):
    """Calculate the Average Directional Movement Index for given data.

    :param df: pandas.DataFrame
    :param n:
    :param n_ADX:
    :return: pandas.DataFrame
    """
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'High'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'Low'], df.loc[i, 'Close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean() / ATR)
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean() / ATR)
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=n_ADX).mean(),
                    name='ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX)
    return df

def adx(df, adxlen=7, dilen=5, return_array=False):
    up = df['High'] - df['High'].shift(1)
    up.iloc[0] = 0
    down = df['Low'].shift(1) - df['Low']
    down.iloc[0] = 0
    plusDM = []
    downDM = []
    for i in range(len(up)):
        plusDM.append(up.iloc[i] if up.iloc[i] > down.iloc[i] and up.iloc[i] > 0  else 0)
        downDM.append(down.iloc[i] if down.iloc[i] > up.iloc[i] and down.iloc[i] > 0 else 0)
    truerange =  np.array(atr(df, dilen))
    plus = np.array(arrayEma(plusDM, dilen)) / truerange * 100
    minus = np.array(arrayEma(downDM, dilen)) / truerange * 100
    su = plus + minus
    su[su==0] = 1
    adx = 100 * np.array(arrayEma(abs(plus - minus) / su, adxlen))
    df['ADX_' + str(adxlen) + '_' + str(dilen)] = adx

    if return_array:
        return df['ADX_' + str(adxlen) + '_' + str(dilen)]

def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference

    :param df: pandas.DataFrame
    :param n_fast:
    :param n_slow:
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def mass_index(df):
    """Calculate the Mass Index for given data.

    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    Range = df['High'] - df['Low']
    EX1 = Range.ewm(span=9, min_periods=9).mean()
    EX2 = EX1.ewm(span=9, min_periods=9).mean()
    Mass = EX1 / EX2
    MassI = pd.Series(Mass.rolling(10).sum(), name='Mass Index')
    df = df.join(MassI)
    return df


def vortex_indicator(df, n):
    """Calculate the Vortex Indicator for given data.

    Vortex Indicator described here:
        http://www.vortexindicator.com/VFX_VORTEX.PDF
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.loc[i + 1, 'High'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'Low'], df.loc[i, 'Close'])
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.loc[i + 1, 'High'] - df.loc[i, 'Low']) - abs(df.loc[i + 1, 'Low'] - df.loc[i, 'High'])
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.Series(VM).rolling(n).sum() / pd.Series(TR).rolling(n).sum(), name='Vortex_' + str(n))
    df = df.join(VI)
    return df


def kst_oscillator(df, r1, r2, r3, r4, n1, n2, n3, n4):
    """Calculate KST Oscillator for given data.

    :param df: pandas.DataFrame
    :param r1:
    :param r2:
    :param r3:
    :param r4:
    :param n1:
    :param n2:
    :param n3:
    :param n4:
    :return: pandas.DataFrame
    """
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(
        ROC1.rolling(n1).sum() + ROC2.rolling(n2).sum() * 2 + ROC3.rolling(n3).sum() * 3 + ROC4.rolling(n4).sum() * 4,
        name='KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(
            n2) + '_' + str(n3) + '_' + str(n4))
    df = df.join(KST)
    return df


def relative_strength_index(df, n, ret=False):
    """Calculate Relative Strength Index(RSI) for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    if ret:
        return RSI
    else:
        df = df.join(RSI)
        return df

def accumulation_distribution(df, n):
    """Calculate Accumulation/Distribution for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))
    df = df.join(AD)
    return df


def chaikin_oscillator(df):
    """Calculate Chaikin Oscillator for given data.

    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    Chaikin = pd.Series(ad.ewm(span=3, min_periods=3).mean() - ad.ewm(span=10, min_periods=10).mean(), name='Chaikin')
    df = df.join(Chaikin)
    return df


def money_flow_index(df, n):
    """Calculate Money Flow Index and Ratio for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.loc[i + 1, 'Volume'])
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(MFR.rolling(n, min_periods=n).mean(), name='MFI_' + str(n))
    df = df.join(MFI)
    return df


def on_balance_volume(df, n):
    """Calculate On-Balance Volume for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] > 0:
            OBV.append(df.loc[i + 1, 'Volume'])
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] == 0:
            OBV.append(0)
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] < 0:
            OBV.append(-df.loc[i + 1, 'Volume'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='=' + str(n))
    df = df.join(OBV_ma)
    return df


def force_index(df, n):
    """Calculate Force Index for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name='Force_' + str(n))
    df = df.join(F)
    return df


def ease_of_movement(df, n):
    """Calculate Ease of Movement for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(EoM.rolling(n, min_periods=n).mean(), name='EoM_' + str(n))
    df = df.join(Eom_ma)
    return df


def commodity_channel_index(df, n):
    """Calculate Commodity Channel Index for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) / PP.rolling(n, min_periods=n).std(),
                    name='CCI_' + str(n))
    df = df.join(CCI)
    return df


def coppock_curve(df, n):
    """Calculate Coppock Curve for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series((ROC1 + ROC2).ewm(span=n, min_periods=n).mean(), name='Copp_' + str(n))
    df = df.join(Copp)
    return df


def keltner_channel(df, n):
    """Calculate Keltner Channel for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    KelChM = pd.Series(((df['High'] + df['Low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChM_' + str(n))
    KelChU = pd.Series(((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChU_' + str(n))
    KelChD = pd.Series(((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChD_' + str(n))
    df = df.join(KelChM)
    df = df.join(KelChU)
    df = df.join(KelChD)
    return df


def ultimate_oscillator(df):
    """Calculate Ultimate Oscillator for given data.

    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'High'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'Low'], df.loc[i, 'Close'])
        TR_l.append(TR)
        BP = df.loc[i + 1, 'Close'] - min(df.loc[i + 1, 'Low'], df.loc[i, 'Close'])
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.Series(BP_l).rolling(7).sum() / pd.Series(TR_l).rolling(7).sum()) + (
                2 * pd.Series(BP_l).rolling(14).sum() / pd.Series(TR_l).rolling(14).sum()) + (
                                 pd.Series(BP_l).rolling(28).sum() / pd.Series(TR_l).rolling(28).sum()),
                     name='Ultimate_Osc')
    df = df.join(UltO)
    return df


def donchian_channel(df, n):
    """Calculate donchian channel of given pandas data frame.
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    dc_l = []
    while i < n - 1:
        dc_l.append(0)
        i += 1

    i = 0
    while i + n - 1 < df.index[-1]:
        dc = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])
        dc_l.append(dc)
        i += 1

    donchian_chan = pd.Series(dc_l, name='Donchian_' + str(n))
    donchian_chan = donchian_chan.shift(n - 1)
    return df.join(donchian_chan)


def standard_deviation(df, n):
    """Calculate Standard Deviation for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    df = df.join(pd.Series(df['Close'].rolling(n, min_periods=n).std(), name='STD_' + str(n)))
    return df
