import pandas as pd
import numpy as np
import pandas_ta as ta # pandas_ta는 여기서 직접 사용되지 않지만, 메인 스크립트에서 ATR 계산 등에 사용될 수 있음
import matplotlib.pyplot as plt # 메인 스크립트에서 직접 사용되지 않음
import scipy # 메인 스크립트에서 직접 사용되지 않음

def plot_two_axes(series1, *ex_series): # 현재 메인 코드에서 사용되지 않음
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
    output = np.full_like(data_array, np.nan) # NaN으로 초기화

    # data_array에서 NaN이 아닌 첫 번째 유효한 값을 찾아 output의 시작점으로 설정
    first_valid_idx = -1
    for i in range(len(data_array)):
        if not np.isnan(data_array[i]):
            output[i] = data_array[i]
            first_valid_idx = i
            break
    
    if first_valid_idx != -1: # 유효한 시작값이 있는 경우에만 계산 진행
        for i in range(first_valid_idx + 1, len(data_array)):
            # 이전 output 값이 NaN인 경우 (이론적으로 발생하면 안 되지만 방어 코드)
            # 또는 현재 data_array 값이 NaN인 경우 처리
            if np.isnan(output[i-1]): # 이전 hawkes 값이 NaN이면, 현재 값으로 시작 (방어적)
                 if not np.isnan(data_array[i]):
                    output[i] = data_array[i]
                 else:
                    output[i] = np.nan # 둘 다 NaN이면 어쩔 수 없이 NaN
            elif np.isnan(data_array[i]): # 현재 입력이 NaN이면 이전 hawkes 값만 감쇠
                output[i] = output[i-1] * alpha
            else: # 둘 다 유효한 값이면 정상 계산
                output[i] = output[i-1] * alpha + data_array[i]
    
    return output * kappa

def hawkes_process(data, kappa):
    """판다스 시리즈 호환 버전 (NaN 처리 강화)"""
    assert(kappa > 0.0)
    alpha = np.exp(-kappa)
    
    if isinstance(data, pd.Series):
        arr = data.to_numpy() # Series를 NumPy 배열로 변환
        output = np.full_like(arr, np.nan) # 결과 배열을 NaN으로 초기화

        # 첫 번째 유효한 (non-NaN) 값으로 output 배열의 시작점 설정
        first_valid_idx = -1
        for i in range(len(arr)):
            if not np.isnan(arr[i]):
                output[i] = arr[i] # hawkes 프로세스의 첫 값은 입력 값 자체로 시작
                first_valid_idx = i
                break
        
        if first_valid_idx != -1: # 유효한 시작값이 있는 경우에만 계산 진행
            for i in range(first_valid_idx + 1, len(arr)):
                # 이전 output 값이 NaN인 경우 (이론적으로 발생하면 안 되지만 방어 코드)
                if np.isnan(output[i-1]):
                    if not np.isnan(arr[i]): # 현재 입력값이 유효하면 그것으로 시작
                        output[i] = arr[i]
                    # else: output[i]는 이미 np.nan으로 초기화됨
                # 현재 입력(arr[i])이 NaN인 경우
                elif np.isnan(arr[i]):
                    output[i] = output[i-1] * alpha # 이전 hawkes 값만 감쇠
                # 모든 값이 유효한 경우
                else:
                    output[i] = output[i-1] * alpha + arr[i]
        
        return pd.Series(output * kappa, index=data.index) # kappa 곱하고 Series로 반환
    
    elif isinstance(data, np.ndarray):
        # NumPy 배열 입력 시 NumPy 버전 직접 호출
        return hawkes_process_np(data, kappa)
    else:
        raise TypeError("Input data must be a pandas Series or a numpy array.")


def vol_signal_np(close_array, vol_hawkes_array, lookback):
    """순수 NumPy 기반 볼륨 신호 계산"""
    length = len(close_array)
    signal = np.zeros(length) # 0으로 초기화 (중립)
    
    if length < lookback: # 데이터가 lookback 기간보다 짧으면 신호 생성 불가
        return signal

    q05 = np.zeros(length)
    q95 = np.zeros(length)
    
    # 롤링 분위수 계산 - 윈도우 방식 사용
    # vol_hawkes_array에 NaN이 없다고 가정 (사전 처리 필요)
    for i in range(lookback -1, length): # lookback-1 부터 시작해야 첫 윈도우가 lookback 길이만큼 참
        window = vol_hawkes_array[i - (lookback - 1) : i + 1]
        if len(window[~np.isnan(window)]) > 0: # 윈도우 내 유효한 값이 있을 경우
            q05[i] = np.nanpercentile(window, 5)
            q95[i] = np.nanpercentile(window, 95)
        else: # 윈도우 전체가 NaN이면 분위수도 NaN
            q05[i] = np.nan
            q95[i] = np.nan

    # lookback 이전 기간은 첫 계산된 분위수 값으로 채우기 (bfill 효과)
    first_q05 = q05[lookback-1]
    first_q95 = q95[lookback-1]
    for i in range(lookback-1):
        q05[i] = first_q05
        q95[i] = first_q95
    
    # NaN이 여전히 남아있다면 0으로 채움 (방어적 코딩)
    q05 = np.nan_to_num(q05, nan=0.0)
    q95 = np.nan_to_num(q95, nan=0.0)

    last_below_idx = -1 # q05 하회한 마지막 인덱스
    curr_sig = 0 # 현재 신호 상태
    
    for i in range(1, length): # 첫 번째 요소는 비교 대상이 없으므로 1부터 시작
        # vol_hawkes_array[i]가 NaN이 아니고, q05[i]도 NaN이 아닐 때 비교
        if not np.isnan(vol_hawkes_array[i]) and not np.isnan(q05[i]):
            if vol_hawkes_array[i] < q05[i]:
                last_below_idx = i
                curr_sig = 0 # 중립으로 리셋
        
        # vol_hawkes_array[i]와 q95[i] 등이 NaN이 아닐 때 비교
        if not np.isnan(vol_hawkes_array[i]) and not np.isnan(q95[i]) and \
           not np.isnan(vol_hawkes_array[i-1]) and not np.isnan(q95[i-1]):
            
            if (vol_hawkes_array[i] > q95[i] and 
                vol_hawkes_array[i-1] <= q95[i-1] and 
                last_below_idx > 0 and i > last_below_idx): # last_below_idx 이후에 발생한 q95 상향 돌파
                
                # close_array[i]와 close_array[last_below_idx]가 NaN이 아닌지 확인
                if not np.isnan(close_array[i]) and not np.isnan(close_array[last_below_idx]):
                    change = close_array[i] - close_array[last_below_idx]
                    if change > 0.0:
                        curr_sig = 1 # 매수
                    else:
                        curr_sig = -1 # 매도 (메인 스크립트에서 0으로 바꿀 수 있음)
                # else: 가격 정보가 NaN이면 신호 변경 없음 (curr_sig 유지)
        
        signal[i] = curr_sig
        
    return signal

def vol_signal(close: pd.Series, vol_hawkes: pd.Series, lookback: int):
    """판다스 시리즈 호환 버전 (내부적으로 vol_signal_np 사용 권장)"""
    # 이 함수는 hawkes.py 내에서 직접 사용되기보다는,
    # 메인 스크립트에서 vol_signal_np를 직접 호출하고 결과를 Series로 변환하는 것이 더 효율적일 수 있습니다.
    # 여기서는 Pandas 기능을 사용한 원래 로직을 유지하되, NaN 처리를 강화합니다.

    if not isinstance(close, pd.Series) or not isinstance(vol_hawkes, pd.Series):
        raise TypeError("Inputs 'close' and 'vol_hawkes' must be pandas Series.")

    if len(close) != len(vol_hawkes):
        raise ValueError("Inputs 'close' and 'vol_hawkes' must have the same length.")

    if len(close) < lookback:
        return pd.Series(np.zeros(len(close)), index=close.index)

    signal = pd.Series(np.zeros(len(close)), index=close.index)
    
    # min_periods=1을 사용하여 롤링 기간 동안 NaN이 아닌 값이 하나라도 있으면 계산
    q05 = vol_hawkes.rolling(window=lookback, min_periods=1).quantile(0.05)
    q95 = vol_hawkes.rolling(window=lookback, min_periods=1).quantile(0.95)

    # 롤링 계산 후 초반 NaN을 뒤의 값으로 채우기 (bfill)
    q05.fillna(method='bfill', inplace=True)
    q95.fillna(method='bfill', inplace=True)
    
    # 그래도 NaN이 남으면 0으로 (방어적)
    q05.fillna(0, inplace=True)
    q95.fillna(0, inplace=True)

    last_below_idx = -1 # Series의 iloc 인덱스 기준
    curr_sig = 0

    for i in range(len(signal)):
        # 모든 비교 대상 값이 유효한지(NaN이 아닌지) 확인
        current_vol_hawk = vol_hawkes.iloc[i]
        current_q05 = q05.iloc[i]
        current_q95 = q95.iloc[i]
        
        if pd.isna(current_vol_hawk) or pd.isna(current_q05) or pd.isna(current_q95):
            signal.iloc[i] = curr_sig # 유효하지 않은 값이 있으면 이전 신호 유지
            continue

        if current_vol_hawk < current_q05:
            last_below_idx = i
            curr_sig = 0

        if i > 0: # i-1 인덱스 접근을 위해
            prev_vol_hawk = vol_hawkes.iloc[i-1]
            prev_q95 = q95.iloc[i-1]
            if pd.isna(prev_vol_hawk) or pd.isna(prev_q95):
                signal.iloc[i] = curr_sig
                continue

            if (current_vol_hawk > current_q95 and 
                prev_vol_hawk <= prev_q95 and 
                last_below_idx != -1 and i > last_below_idx):
                
                current_close_price = close.iloc[i]
                last_below_close_price = close.iloc[last_below_idx]

                if pd.isna(current_close_price) or pd.isna(last_below_close_price):
                    signal.iloc[i] = curr_sig # 가격 정보 NaN이면 이전 신호 유지
                    continue
                
                change = current_close_price - last_below_close_price
                if change > 0.0:
                    curr_sig = 1
                else:
                    curr_sig = -1 # 메인 스크립트에서 0으로 바꿀 수 있음
        signal.iloc[i] = curr_sig

    return signal


def get_trades_from_signal(data: pd.DataFrame, signal_series: pd.Series): # signal 타입을 Series로 변경
    # Gets trade entry and exit times from a signal Series
    # that has values of -1, 0, 1. Denoting short,flat,and long.
    # No position sizing.

    long_trades = []
    short_trades = []

    # DataFrame과 Series의 인덱스가 동일하다고 가정
    if not data.index.equals(signal_series.index):
        raise ValueError("DataFrame index and signal Series index do not match.")

    close_arr = data['close'].to_numpy()
    signal_arr = signal_series.to_numpy() # Series를 NumPy 배열로 변환
    
    last_sig = 0.0
    open_trade = None # [entry_time, entry_price, exit_time, exit_price]
    idx = data.index # Pandas DateTimeIndex

    for i in range(len(data)):
        current_sig = signal_arr[i]
        current_time = idx[i]
        current_price = close_arr[i]

        if pd.isna(current_sig) or pd.isna(current_price): # 신호 또는 가격이 NaN이면 건너뜀
            if open_trade is not None and last_sig != 0: # 포지션이 있는데 현재 정보가 NaN이면 강제 청산 고려 가능 (여기선 생략)
                pass
            last_sig = current_sig if not pd.isna(current_sig) else last_sig # NaN이 아닌 경우에만 last_sig 업데이트
            continue


        # 롱 진입: 이전 신호가 1이 아니었고 현재 신호가 1일 때
        if current_sig == 1.0 and last_sig != 1.0:
            if open_trade is not None: # 기존 포지션(숏) 청산
                if last_sig == -1.0: # 숏 포지션이었다면
                    open_trade[2] = current_time
                    open_trade[3] = current_price
                    short_trades.append(open_trade)
                # else: 롱 포지션 중 재진입 또는 중립에서 롱 진입 (이 경우는 아래에서 처리)
            open_trade = [current_time, current_price, -1, np.nan] # 새 롱 포지션 시작

        # 숏 진입: 이전 신호가 -1이 아니었고 현재 신호가 -1일 때
        elif current_sig == -1.0 and last_sig != -1.0:
            if open_trade is not None: # 기존 포지션(롱) 청산
                if last_sig == 1.0: # 롱 포지션이었다면
                    open_trade[2] = current_time
                    open_trade[3] = current_price
                    long_trades.append(open_trade)
            open_trade = [current_time, current_price, -1, np.nan] # 새 숏 포지션 시작
        
        # 포지션 청산: 현재 신호가 0이고 이전 신호가 0이 아니었을 때
        elif current_sig == 0.0 and last_sig != 0.0:
            if open_trade is not None:
                open_trade[2] = current_time
                open_trade[3] = current_price
                if last_sig == 1.0: # 롱 포지션 청산
                    long_trades.append(open_trade)
                elif last_sig == -1.0: # 숏 포지션 청산
                    short_trades.append(open_trade)
                open_trade = None

        last_sig = current_sig

    # 루프 종료 후 열려있는 포지션이 있다면 마지막 데이터로 청산 (선택 사항)
    # if open_trade is not None and open_trade[2] == -1: # exit_time이 설정되지 않았다면
    #     open_trade[2] = idx[-1]
    #     open_trade[3] = close_arr[-1]
    #     if last_sig == 1.0:
    #         long_trades.append(open_trade)
    #     elif last_sig == -1.0:
    #         short_trades.append(open_trade)

    long_trades_df = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades_df = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    if not long_trades_df.empty:
        long_trades_df['percent'] = (long_trades_df['exit_price'] - long_trades_df['entry_price']) / long_trades_df['entry_price']
        long_trades_df = long_trades_df.set_index('entry_time')
    
    if not short_trades_df.empty:
        short_trades_df['percent'] = -1 * (short_trades_df['exit_price'] - short_trades_df['entry_price']) / short_trades_df['entry_price']
        short_trades_df = short_trades_df.set_index('entry_time')
        
    return long_trades_df, short_trades_df


