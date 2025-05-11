import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import scipy

def plot_two_axes(series1, *ex_series):
    plt.style.use('dark_background')
    ax = series1.plot(color='green')
    ax2 = ax.twinx()
    for i, series in enumerate(ex_series):
        series.plot(ax=ax2, alpha=0.5)
    #plt.show()

def hawkes_process_np(data_array, kappa):
    """순수 NumPy 기반 호크스 프로세스 계산"""
    assert(kappa > 0.0)
    alpha = np.exp(-kappa)
    output = np.zeros(len(data_array))
    output[0] = data_array[0] if not np.isnan(data_array[0]) else 0  # 첫 번째 요소 초기화
    
    for i in range(1, len(data_array)):
        if np.isnan(data_array[i]):
            output[i] = output[i-1] * alpha
        else:
            output[i] = output[i-1] * alpha + data_array[i]
    
    return output * kappa

def hawkes_process(data, kappa):
    """판다스 시리즈 호환 버전"""
    assert(kappa > 0.0)
    alpha = np.exp(-kappa)
    
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
        output = np.zeros(len(data))
        output[:] = np.nan
        for i in range(1, len(data)):
            if np.isnan(output[i - 1]):
                output[i] = arr[i]
            else:
                output[i] = output[i - 1] * alpha + arr[i]
        return pd.Series(output, index=data.index) * kappa
    else:
        # NumPy 배열 입력 시 NumPy 버전 호출
        return hawkes_process_np(data, kappa)

def vol_signal_np(close_array, vol_hawkes_array, lookback):
    """순수 NumPy 기반 볼륨 신호 계산"""
    length = len(close_array)
    signal = np.zeros(length)
    
    # 롤링 분위수 계산 - 윈도우 방식 사용
    q05 = np.zeros(length)
    q95 = np.zeros(length)
    
    for i in range(lookback, length):
        window = vol_hawkes_array[i-lookback:i]
        q05[i] = np.percentile(window, 5)
        q95[i] = np.percentile(window, 95)
    
    # 이전 값으로 NaN 채우기
    for i in range(1, lookback):
        q05[i] = q05[lookback]
        q95[i] = q95[lookback]
    
    # 신호 생성
    last_below = -1
    curr_sig = 0
    
    for i in range(1, length):
        if vol_hawkes_array[i] < q05[i]:
            last_below = i
            curr_sig = 0
            
        if (vol_hawkes_array[i] > q95[i] and 
            vol_hawkes_array[i-1] <= q95[i-1] and 
            last_below > 0):
            
            change = close_array[i] - close_array[last_below]
            if change > 0.0:
                curr_sig = 1
            else:
                curr_sig = -1
                
        signal[i] = curr_sig
        
    return signal

def vol_signal(close, vol_hawkes, lookback):
    signal = np.zeros(len(close))
    q05 = vol_hawkes.rolling(lookback).quantile(0.05)
    q95 = vol_hawkes.rolling(lookback).quantile(0.95)
    
    last_below = -1
    curr_sig = 0

    for i in range(len(signal)):
        if vol_hawkes.iloc[i] < q05.iloc[i]:
            last_below = i
            curr_sig = 0

        if vol_hawkes.iloc[i] > q95.iloc[i] \
           and vol_hawkes.iloc[i - 1] <= q95.iloc[i - 1] \
           and last_below > 0 :
            
            change = close.iloc[i] - close.iloc[last_below]
            if change > 0.0:
                curr_sig = 1
            else:
                curr_sig = -1
        signal[i] = curr_sig

    return signal

def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
    # Gets trade entry and exit times from a signal
    # that has values of -1, 0, 1. Denoting short,flat,and long.
    # No position sizing.

    long_trades = []
    short_trades = []

    close_arr = data['close'].to_numpy()
    last_sig = 0.0
    open_trade = None
    idx = data.index
    for i in range(len(data)):
        if signal[i] == 1.0 and last_sig != 1.0: # Long entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]
        if signal[i] == -1.0  and last_sig != -1.0: # Short entry
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]
        
        if signal[i] == 0.0 and last_sig == -1.0: # Short exit
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            short_trades.append(open_trade)
            open_trade = None

        if signal[i] == 0.0  and last_sig == 1.0: # Long exit
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            long_trades.append(open_trade)
            open_trade = None

        last_sig = signal[i]

    long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    long_trades['percent'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price']
    short_trades['percent'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
    long_trades = long_trades.set_index('entry_time')
    short_trades = short_trades.set_index('entry_time')
    return long_trades, short_trades
