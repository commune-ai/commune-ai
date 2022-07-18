#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:45:05 2019

@author: devinreich
"""
import numpy as np
import pandas as pd

def rma(source, length):
    alpha = length
    s = [0]
    for i in range(1, len(source)):
        s.append( (source[i] + (alpha - 1) * s[i-1]) / alpha )
    return s
        
def sma(source, length):
    s = [0] * (length - 1)
    for i in range(length, len(source) + 1):
        s.append( sum(source[i-length:i]) / length)
    return s

def ema(source, length):
    m = (2 / (length + 1))
    f = sum(source[:length]) / length
    emas = [0] * (length - 1)
    emas.append(f)
    for i in range(length, len(source)):
        emas.append((source[i] -source[i - 1]) * m + (source[i - 1]))
    return emas

def wma(source, length):
    wmas = [0] * (length - 1)
    for i in range(length, len(source)):
        s = 0
        w = 0
        for j in range(1,length + 1):
            s = s + (source[i - length + j - 1] * j)
            w = w + j
        wmas.append(s/w)
    return wmas

def highest(source, length):
    highs = [source[0]]
    for i in range(2, length):
        highs.append(max(source[0:i]))
    for i in range(length, len(source) + 1):
        highs.append(max(source[i-length:i]))
    return highs
        
def lowest(source, length):
    lows = [source[0]]
    for i in range(2, length):
        lows.append(min(source[0:i]))
    for i in range(length, len(source) + 1):
        lows.append(min(source[i-length:i]))
    return lows

# TODO adapt to handle length as a sequence
def rsi(source, length):
    u = [0]
    d = [0]
    for i in range(1, len(source)):
        u.append(max(source[i] - source[i-1], 0))
        d.append(max(source[i-1] - source[i], 0))
    rs = np.array(rma(u, length)) / np.array(rma(d, length))
    res = []
    for i in range(len(rs)):
        res.append(100 - (100 / (1 + rs[i])))
        
    return res

def std(source, length):
    std = []
    for i in range(1, length):
        std.append(np.std(source[0:i]))
    for i in range(length, len(source) + 1):
        std.append(np.std(source[i-length:i]))
    return std

def true_range(df):
    tr = [0]
    for i in range(1, len(df)):
        tr.append(max(df['High'].iloc[i] - df['Low'].iloc[i],
                  abs(df['High'].iloc[i] - df['Close'].iloc[i-1]),
                  abs(df['Low'].iloc[i] - df['Close'].iloc[i-1])))
    return tr
    
def pine_sum(source, length):
    s = []
    for i in range(1, length):
        s.append(sum(np.round(np.array(source[0:i]), decimals=7)))
    for i in range(length, len(source)+1):
        s.append(sum(np.round(np.array(source[i-length:i]), decimals=7)))
    return s

def change(source, length=1):
    c = [0] * length
    for i in range(length, len(source)):
        c.append(source[i] - source[i-1])
    return c

def dev(source, length):
    return np.array(source) - np.array(sma(source,length))

# TODO adapt to handle other argument inputs
def pine_max(source, value):
    ms = []
    for i in range(len(source)):
        ms.append(max(source[i],value))
    return ms

# TODO adapt to handle other argument inputs
def pine_min(source, value):
    ms = []
    for i in range(len(source)):
        ms.append(min(source[i],value))
    return ms
    
def crossover(source, value):
    cross = [False]
    if type(value) == list or type(value) == np.ndarray:
        for i in range(1, len(source)):
            cross.append(source[i] > value[i] and source[i-1] < value[i-1])
    else:
        for i in range(1, len(source)):
            cross.append(source[i] > value and source[i-1] < value)
    return cross

def crossunder(source, value):

    cross = [False]
    if type(value) == list or type(value) == np.ndarray:
        for i in range(1, len(source)):
            cross.append(source[i] < value[i] and source[i-1] > value[i-1])
    else:
        for i in range(1, len(source)):
            cross.append(source[i] < value and source[i-1] > value)
    return cross

def cross(source, source2):
    co = np.array(crossover(source,source2))
    cu = np.array(crossunder(source, source2))
    cross = []
    for i in range(len(cu)):
        cross.append(cu[i] or co[i])
    return cross

def rising(source, length):
    s = np.array(source)
    rise = [False] * length
    for i in range(length, len(s)):
        rise.append((s[i] > s[i-length:i]).any())
    return rise

def falling(source, length):
    s = np.array(source)
    fall = [False] * length
    for i in range(length, len(s)):
        fall.append((s[i] < s[i-length:i]).any())
    return fall

def sar(df, mx=0.2, acc=0.02):
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
        
    return psarDF['PSAR'].values

def stoch(source, high, low, length):
    lows = lowest(low, length)
    highs = highest(high, length)
    stoch = []
    for i in range(len(source)):
        stoch.append(100* (source[i] - lows[i]) / (highs[i] - lows[i]))
    return stoch

def pine_ema(source, length):
    alpha = 2 / (length + 1)
    s = [alpha * source[0]]
    for i in range(1, len(source)):
        s.append(alpha * source[i] + (1 - alpha) * s[i-1])
    return s

def atr(df, length):
    tr= true_range(df)
    return rma(tr, length)

def percent_change(source, period=1):
    change = [0] * period
    for i in range(period, len(source)):
        change.append((source[i] - source[i-period]) / source[i-period])
    return np.array(change)