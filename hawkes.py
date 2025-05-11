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

def hawkes_process(data: pd.Series, kappa: float):
    assert(kappa > 0.0)
    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    output = np.zeros(len(data))
    output[:] = np.nan
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    return pd.Series(output, index=data.index) * kappa

def vol_signal(close: pd.Series, vol_hawkes: pd.Series, lookback:int):
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

# 새로 추가된 안전한 버전의 vol_signal 함수
def vol_signal_safe(close: pd.Series, vol_hawkes: pd.Series, lookback: int):
    """인덱스 안정성이 보장된 호크스 신호 생성 함수"""
    # 데이터 길이 확인 및 인덱스 일치 확인
    if len(close) != len(vol_hawkes) or not close.index.equals(vol_hawkes.index):
        print("vol_signal_safe: 입력 시리즈의 길이 또는 인덱스가 일치하지 않습니다.")
        # 안전한 기본값 반환 (중립 신호)
        return pd.Series(0, index=close.index)
    
    # 결과 시리즈 초기화 (인덱스 보존)
    signal = pd.Series(0, index=close.index)
    q05 = vol_hawkes.rolling(lookback).quantile(0.05)
    q95 = vol_hawkes.rolling(lookback).quantile(0.95)
    
    # NaN 값 처리
    q05 = q05.fillna(method='bfill').fillna(method='ffill')
    q95 = q95.fillna(method='bfill').fillna(method='ffill')
    
    # 인덱스 안전성을 위해 데이터프레임 사용
    df = pd.DataFrame({
        'close': close,
        'vol_hawkes': vol_hawkes,
        'q05': q05,
        'q95': q95,
        'signal': signal
    })
    
    # 마지막으로 5% 분위수 이하 값을 가진 인덱스 추적
    last_below_idx = None
    curr_sig = 0
    
    # 날짜 순서대로 처리하기 위해 인덱스 정렬
    df = df.sort_index()
    
    # 각 행에 대해 신호 생성 로직 적용
    for idx, row in df.iterrows():
        # 5% 분위수 이하로 떨어진 경우
        if row['vol_hawkes'] < row['q05']:
            last_below_idx = idx
            curr_sig = 0
        
        # 95% 분위수 위로 올라간 경우 (크로스오버)
        if (row['vol_hawkes'] > row['q95'] and 
            last_below_idx is not None):
            
            # 이전 인덱스 찾기 (안전하게)
            try:
                prev_idx = df.index[df.index.get_loc(idx) - 1]
                if prev_idx in df.index and df.loc[prev_idx, 'vol_hawkes'] <= df.loc[prev_idx, 'q95']:
                    # 이전 저점 대비 가격 변화 방향 확인
                    if last_below_idx in df.index:
                        change = row['close'] - df.loc[last_below_idx, 'close']
                        if change > 0.0:
                            curr_sig = 1
                        else:
                            curr_sig = -1
            except (KeyError, IndexError) as e:
                print(f"인덱스 처리 중 오류: {e}")
                # 오류 발생 시 현재 신호 유지
        
        # 현재 신호 저장
        df.at[idx, 'signal'] = curr_sig
    
    return df['signal']

# 다음 코드 블록은 독립적인 테스트용이므로 주석 처리하여 실제 애플리케이션 실행에 영향을 주지 않도록 합니다
'''
data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

# Normalize volume
norm_lookback = 336
data['atr'] = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), norm_lookback) 
data['norm_range'] = (np.log(data['high']) - np.log(data['low'])) / data['atr']
#plot_two_axes(np.log(data['close']), data['norm_range'])

data['v_hawk'] = hawkes_process(data['norm_range'], 0.1)
data['sig'] = vol_signal(data['close'], data['v_hawk'], 168)

data['next_return'] = np.log(data['close']).diff().shift(-1)
data['signal_return'] = data['sig'] * data['next_return']
win_returns = data[data['signal_return'] > 0]['signal_return'].sum()
lose_returns = data[data['signal_return'] < 0]['signal_return'].abs().sum()
signal_pf = win_returns / lose_returns
plt.style.use('dark_background')
data['signal_return'].cumsum().plot()

long_trades, short_trades = get_trades_from_signal(data, data['sig'].to_numpy())
long_win_rate = len(long_trades[long_trades['percent'] > 0]) / len(long_trades)
short_win_rate = len(short_trades[short_trades['percent'] > 0]) / len(short_trades)
long_average = long_trades['percent'].mean()
short_average = short_trades['percent'].mean()
time_in_market = len(data[data['sig'] != 0.0]) / len(data)

print("Profit Factor", signal_pf)
print("Long Win Rate", long_win_rate) 
print("Long Average", long_average) 
print("Short Win Rate", short_win_rate) 
print("Short Average", short_average)
print("Time In Market", time_in_market)

# Code for the heatmap
kappa_vals = [0.5, 0.25, 0.1, 0.05, 0.01]
lookback_vals = [24, 48, 96, 168, 336] 
pf_df = pd.DataFrame(index=lookback_vals, columns=kappa_vals)

for lb in lookback_vals:
    for k in kappa_vals:
        data['v_hawk'] = hawkes_process(data['norm_range'], k)
        data['sig'] = vol_signal(data['close'], data['v_hawk'], lb)

        data['next_return'] = np.log(data['close']).diff().shift(-1)
        data['signal_return'] = data['sig'] * data['next_return']
        win_returns = data[data['signal_return'] > 0]['signal_return'].sum()
        lose_returns = data[data['signal_return'] < 0]['signal_return'].abs().sum()
        signal_pf = win_returns / lose_returns

        pf_df.loc[lb, k] = float(signal_pf)
    
plt.style.use('dark_background')
import seaborn as sns
pf_df = pf_df.astype(float)
sns.heatmap(pf_df, annot=True, fmt='f')
plt.xlabel('Hawkes Kappa')
plt.ylabel('Threshold Lookback')
'''
