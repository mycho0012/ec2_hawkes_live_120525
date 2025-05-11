import os
import time
import datetime
import numpy as np
import pandas as pd
import json
import logging
import argparse
import traceback # traceback 추가
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import pyupbit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# 웹서버 기능 추가를 위한 모듈
import threading
import http.server
import socketserver
import socket

# pandas_ta 패치 적용 (Linux 호환성 문제 해결)
def fix_pandas_ta_on_load(): # 함수 이름 변경
    """pandas_ta 패치 자동 적용"""
    import os
    import sys
    
    # site-packages 및 dist-packages 경로 검색
    paths_to_check = [p for p in sys.path if 'site-packages' in p or 'dist-packages' in p]
    
    # 홈 디렉토리의 로컬 Python 경로도 검색 (더 많은 버전 고려)
    home = os.path.expanduser('~')
    for py_ver_minor in range(7, 13): # Python 3.7 ~ 3.12
        local_path = os.path.join(home, '.local', 'lib', f'python3.{py_ver_minor}', 'site-packages')
        if os.path.exists(local_path) and local_path not in paths_to_check:
            paths_to_check.append(local_path)

    for site_pkg_path in paths_to_check:
        squeeze_path = os.path.join(site_pkg_path, 'pandas_ta', 'momentum', 'squeeze_pro.py')
        if os.path.exists(squeeze_path):
            try:
                with open(squeeze_path, 'r', encoding='utf-8') as f: # encoding 명시
                    content = f.read()
                
                if 'from numpy import NaN as npNaN' in content:
                    fixed_content = content.replace('from numpy import NaN as npNaN', 'from numpy import nan as npNaN')
                    with open(squeeze_path, 'w', encoding='utf-8') as f: # encoding 명시
                        f.write(fixed_content)
                    print(f"pandas_ta 패치 적용 완료: {squeeze_path}")
                    logging.info(f"pandas_ta 패치 적용 완료: {squeeze_path}")
                    return True
            except Exception as e:
                print(f"pandas_ta 패치 적용 중 오류 ({squeeze_path}): {str(e)}")
                logging.warning(f"pandas_ta 패치 적용 중 오류 ({squeeze_path}): {str(e)}")
    print("pandas_ta 패치 대상 파일을 찾지 못했거나 이미 패치되었을 수 있습니다.")
    logging.info("pandas_ta 패치 대상 파일을 찾지 못했거나 이미 패치되었을 수 있습니다.")
    return False

# 패치 적용 시도 (스크립트 로드 시 1회)
fix_pandas_ta_on_load()

# 이제 안전하게 pandas_ta 임포트
import pandas_ta as ta
from hawkes import hawkes_process, vol_signal_np # hawkes_process_np는 hawkes_process 내부에서 사용

# 로깅 설정
logging.basicConfig(
    filename='ec2_hawkes_live.log', # 로그 파일명
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 환경 변수 로드 (.env 파일에 API 키 저장)
load_dotenv()

# 명령줄 인자 파싱
parser = argparse.ArgumentParser(description='EC2 Hawkes 프로세스 트레이딩 봇')
parser.add_argument('--kappa', type=float, default=0.3, help='호크스 프로세스 감쇠 계수 (기본값: 0.3)')
parser.add_argument('--lookback', type=int, default=72, help='변동성 기준 룩백 기간 (기본값: 72)')
args = parser.parse_args()

# Upbit API 키 설정 (환경 변수 또는 기본값)
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY') # 기본값 제거, .env 필수
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY') # 기본값 제거, .env 필수

# Slack API 설정
SLACK_API_TOKEN = os.getenv('SLACK_API_TOKEN')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL')

# 웹서버 설정
HOST_IP = os.getenv('HOST_IP', '')
WEB_PORT = int(os.getenv('WEB_PORT', '8500'))

# 거래 설정
TICKER = "KRW-BTC"
CANDLE_INTERVAL = "minute60" # 1시간 캔들
LOOKBACK_HOURS = 2000 # 데이터 수집 기간

# 파라미터 설정
try:
    env_kappa = os.getenv('KAPPA')
    KAPPA = float(env_kappa) if env_kappa is not None else args.kappa
except (ValueError, TypeError):
    KAPPA = args.kappa
    logging.warning(f"KAPPA 환경변수 파싱 오류. 명령줄 인자 또는 기본값 사용: {KAPPA}")

try:
    env_lookback = os.getenv('VOLATILITY_LOOKBACK')
    VOLATILITY_LOOKBACK = int(env_lookback) if env_lookback is not None else args.lookback
except (ValueError, TypeError):
    VOLATILITY_LOOKBACK = args.lookback
    logging.warning(f"VOLATILITY_LOOKBACK 환경변수 파싱 오류. 명령줄 인자 또는 기본값 사용: {VOLATILITY_LOOKBACK}")

COMMISSION_RATE = 0.0005

class EC2HawkesTrader:
    def __init__(self):
        self.upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.slack_client = WebClient(token=SLACK_API_TOKEN) if SLACK_API_TOKEN and SLACK_CHANNEL else None
        self.slack_channel = SLACK_CHANNEL
        
        self.http_port = WEB_PORT
        self.server_thread = None
        self.charts_url_base = None
        self.httpd = None # httpd 인스턴스 변수 추가
        self.start_http_server()
        
        self.current_position = 0
        self.position_entry_price = 0
        self.position_entry_time = None
        self.trading_data = pd.DataFrame() # DataFrame으로 직접 관리
        self.last_signal = 0 # 초기 신호는 중립(0)
        
        self.trade_history = []
        self.num_trades = 0
        
        self.charts_dir = os.path.join(os.getcwd(), 'charts')
        os.makedirs(self.charts_dir, exist_ok=True)
        
        self.send_to_slack(f"🚀 EC2 Hawkes 트레이딩 봇 시작 (KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK})")
        self.load_initial_data()

    def start_http_server(self):
        """간단한 HTTP 서버 시작"""
        try:
            # 호스트 IP 가져오기
            effective_host_ip = HOST_IP
            if not effective_host_ip:
                try:
                    hostname = socket.gethostname()
                    effective_host_ip = socket.gethostbyname(hostname)
                except socket.gaierror:
                    effective_host_ip = "127.0.0.1" # Fallback
                    logging.warning(f"호스트 IP 자동 감지 실패. {effective_host_ip} 사용.")

            # 현재 작업 디렉토리로 이동 (charts 디렉토리 서빙을 위함)
            # SimpleHTTPRequestHandler는 실행된 디렉토리 기준으로 파일을 찾으므로,
            # charts 디렉토리 자체를 서빙하려면 os.chdir(self.charts_dir)을 하거나,
            # 핸들러를 커스터마이징 해야 함. 여기서는 현재 디렉토리에서 charts/ 하위로 접근.
            
            # HTTP 서버 핸들러 설정
            # Python 3.7+ 에서는 http.server.SimpleHTTPRequestHandler의 directory 파라미터 사용 가능
            if sys.version_info >= (3, 7):
                handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=os.getcwd(), **kwargs)
            else: # 구버전 호환 (이 경우 charts 디렉토리로 chdir 하거나 URL에 /charts/ 명시 필요)
                handler = http.server.SimpleHTTPRequestHandler


            self.httpd = socketserver.TCPServer(("", self.http_port), handler)
            self.charts_url_base = f"http://{effective_host_ip}:{self.http_port}/charts" # URL에 /charts/ 명시
            
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()
            
            log_msg = f"HTTP 서버 시작: http://{effective_host_ip}:{self.http_port} (차트 URL 기반: {self.charts_url_base})"
            logging.info(log_msg)
            self.send_to_slack(f"📡 차트 조회 서버 시작: {self.charts_url_base}")
        except Exception as e:
            error_msg = f"HTTP 서버 시작 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            self.send_to_slack(f"❌ {error_msg}")
    
    def send_to_slack(self, message):
        if not self.slack_client or not self.slack_channel:
            logging.info(f"Slack 미설정 - 메시지: {message}")
            return
        try:
            response = self.slack_client.chat_postMessage(channel=self.slack_channel, text=message)
            logging.info(f"Slack 메시지 전송됨: {message[:100]}...") # 긴 메시지 로그 축약
        except SlackApiError as e:
            logging.error(f"Slack 메시지 전송 오류 (API): {e.response['error']}")
        except Exception as e:
            logging.error(f"Slack 메시지 전송 중 일반 오류: {str(e)}")
            logging.error(traceback.format_exc())

    def load_initial_data(self):
        """초기 데이터 로드 및 지표 계산 (DataFrame 직접 사용)"""
        try:
            logging.info("초기 데이터 로드 중...")
            self.send_to_slack("📊 초기 데이터 로드 중...")
            
            df = pyupbit.get_ohlcv(TICKER, interval=CANDLE_INTERVAL, count=LOOKBACK_HOURS, period=1) # period=1 추가
            if df is None or df.empty:
                raise Exception("Upbit에서 데이터를 가져오지 못했습니다.")

            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()
            
            self.trading_data = df.copy()
            
            self._calculate_and_add_indicators()
            
            if not self.trading_data.empty and 'signal' in self.trading_data.columns:
                self.last_signal = int(self.trading_data['signal'].iloc[-1])
            else:
                self.last_signal = 0 # 데이터 없거나 signal 없으면 중립
                logging.warning("초기 데이터 로드 후 trading_data가 비어있거나 signal 열이 없습니다.")
            
            log_msg = f"초기 데이터 로드 완료: {len(self.trading_data)} 개의 캔들"
            logging.info(log_msg)
            self.send_to_slack(f"✅ {log_msg}")
            
            self.calculate_performance_metrics()
            
            log_msg = f"사용 파라미터 - KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK}"
            logging.info(log_msg)
            self.send_to_slack(f"⚙️ {log_msg}")
            
            self.log_account_info()
            
        except Exception as e:
            error_msg = f"초기 데이터 로드 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(f"스택 트레이스: {traceback.format_exc()}")
            self.send_to_slack(f"❌ {error_msg}")
            # 초기 로드 실패 시 프로그램 종료 또는 재시도 로직 필요할 수 있음
            raise # 일단 에러 발생시키고 종료

    def _calculate_and_add_indicators(self):
        """DataFrame에 직접 호크스 프로세스 관련 지표를 계산하고 추가합니다."""
        if self.trading_data.empty:
            logging.warning("지표 계산을 위한 데이터가 없습니다.")
            # 필요한 모든 열을 NaN 또는 0으로 채워넣어 이후 코드 실행 보장
            cols = ['log_high', 'log_low', 'log_close', 'atr', 'norm_range', 
                    'v_hawk', 'q05', 'q95', 'signal']
            for col in cols:
                if col not in self.trading_data.columns:
                    self.trading_data[col] = np.nan if col != 'signal' else 0
            return

        td = self.trading_data # 편의를 위한 참조

        try:
            td['log_high'] = np.log(td['high'])
            td['log_low'] = np.log(td['low'])
            td['log_close'] = np.log(td['close'])

            atr_series = ta.atr(td['log_high'], td['log_low'], td['log_close'], length=336) # ATR 계산 기간 확인 필요
            if atr_series is None: # pandas_ta가 None을 반환하는 경우 대비
                atr_series = pd.Series(np.nan, index=td.index)
            td['atr'] = atr_series
            td['atr'].fillna(method='ffill', inplace=True)
            td['atr'].fillna(method='bfill', inplace=True)
            td['atr'].fillna(0, inplace=True) # 모든 NaN을 0으로 (0으로 나누기 방지 위함)

            # ATR이 0이거나 매우 작은 경우를 대비하여 분모에 작은 값(epsilon) 추가
            epsilon = 1e-9 
            td['norm_range'] = (td['log_high'] - td['log_low']) / (td['atr'] + epsilon)
            td['norm_range'].replace([np.inf, -np.inf], 0, inplace=True)
            td['norm_range'].fillna(0, inplace=True)

            # hawkes_process 입력 전 norm_range의 NaN 최종 확인
            if td['norm_range'].isnull().any():
                logging.warning("norm_range에 NaN 값이 남아있습니다. 0으로 채웁니다.")
                td['norm_range'].fillna(0, inplace=True)

            v_hawk_series = hawkes_process(td['norm_range'], KAPPA)
            if v_hawk_series is None: # hawkes_process가 None을 반환하는 경우 대비
                 v_hawk_series = pd.Series(np.nan, index=td.index)
            td['v_hawk'] = v_hawk_series
            td['v_hawk'].fillna(method='ffill', inplace=True)
            td['v_hawk'].fillna(method='bfill', inplace=True)
            td['v_hawk'].fillna(0, inplace=True) # 최종 NaN 처리

            # v_hawk에 NaN이 없도록 확실히 한 후 롤링 계산
            if td['v_hawk'].isnull().any():
                logging.error("v_hawk 계산 후에도 NaN이 남아있습니다. 0으로 강제 채움.")
                td['v_hawk'].fillna(0, inplace=True)

            td['q05'] = td['v_hawk'].rolling(window=VOLATILITY_LOOKBACK, min_periods=1).quantile(0.05)
            td['q95'] = td['v_hawk'].rolling(window=VOLATILITY_LOOKBACK, min_periods=1).quantile(0.95)
            td['q05'].fillna(method='bfill', inplace=True)
            td['q95'].fillna(method='bfill', inplace=True)
            td['q05'].fillna(0, inplace=True) # 최종 NaN 처리
            td['q95'].fillna(0, inplace=True) # 최종 NaN 처리

            # vol_signal_np 입력 전 close와 v_hawk의 NaN 최종 확인
            close_np = td['close'].fillna(0).to_numpy() # NaN이면 0으로
            v_hawk_np = td['v_hawk'].fillna(0).to_numpy() # NaN이면 0으로

            signal_np_array = vol_signal_np(close_np, v_hawk_np, VOLATILITY_LOOKBACK)
            td['signal'] = pd.Series(signal_np_array, index=td.index)
            
            td.loc[td['signal'] < 0, 'signal'] = 0 # 롱 온리
            td['signal'].fillna(0, inplace=True) # 신호의 NaN은 0(중립)으로

            logging.info("지표 계산 및 DataFrame 업데이트 완료.")

        except Exception as e:
            error_msg = f"지표 계산 중 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(f"스택 트레이스: {traceback.format_exc()}")
            # 오류 발생 시 필요한 열들이 없을 수 있으므로 안전하게 초기화
            cols_to_ensure = ['log_high', 'log_low', 'log_close', 'atr', 'norm_range', 
                              'v_hawk', 'q05', 'q95', 'signal']
            for col_ in cols_to_ensure:
                if col_ not in td.columns:
                    td[col_] = np.nan if col_ != 'signal' else 0
            if 'signal' in td.columns: td['signal'].fillna(0, inplace=True) # signal은 0으로
            self.send_to_slack(f"❌ {error_msg}")
    
    def update_data(self):
        """데이터 업데이트 (DataFrame 직접 사용)"""
        try:
            logging.info("데이터 업데이트 시작...")
            max_retries = 3
            retry_count = 0
            fresh_data_df = None
            
            while retry_count < max_retries:
                try:
                    # period=1: API 요청시 딜레이를 줄 수 있음. Upbit API wrapper 확인 필요.
                    fresh_data_df = pyupbit.get_ohlcv(TICKER, interval=CANDLE_INTERVAL, count=LOOKBACK_HOURS, period=0.2) 
                    if fresh_data_df is not None and not fresh_data_df.empty:
                        break
                except Exception as retry_e:
                    retry_count += 1
                    logging.warning(f"OHLCV 데이터 가져오기 재시도 {retry_count}/{max_retries}: {str(retry_e)}")
                    time.sleep(10) # 재시도 간격 증가
            
            if fresh_data_df is None or fresh_data_df.empty:
                # 데이터를 가져오지 못했을 경우, 이전 데이터를 유지하고 경고 로깅
                logging.error("캔들 데이터를 가져오지 못했습니다. 이전 데이터를 사용하여 계속 진행합니다.")
                self.send_to_slack("⚠️ 캔들 데이터를 가져오지 못했습니다. 이전 데이터로 계속 시도합니다.")
                # 이 경우 _calculate_and_add_indicators()를 호출하지 않거나,
                # 이전 데이터에 대해 다시 호출할지 결정해야 함. 여기서는 일단 이전 데이터 유지.
                if self.trading_data.empty: # 이전 데이터조차 없다면 심각한 문제
                     raise Exception("초기 데이터 로드도 실패했고, 업데이트도 실패했습니다.")
                # 이전 데이터가 있다면, 그것에 대해 지표 재계산 (선택적)
                # self._calculate_and_add_indicators() 
                return # 업데이트 실패로 간주하고 다음 사이클로

            fresh_data_df = fresh_data_df[~fresh_data_df.index.duplicated(keep='last')]
            fresh_data_df = fresh_data_df.sort_index()
            
            self.trading_data = fresh_data_df.copy()
            
            self._calculate_and_add_indicators()
            
            if not self.trading_data.empty and 'signal' in self.trading_data.columns and 'close' in self.trading_data.columns:
                self.last_signal = int(self.trading_data['signal'].iloc[-1])
                
                last_candle = self.trading_data.iloc[-1]
                date_str = last_candle.name.strftime('%Y-%m-%d %H:%M')
                # .get(key, np.nan)을 사용하여 키가 없는 경우에도 에러 방지
                candle_info = (
                    f"📈 새 캔들 업데이트 ({date_str})\n"
                    f"가격: {last_candle.get('close', np.nan):,.0f} KRW (고가: {last_candle.get('high', np.nan):,.0f}, 저가: {last_candle.get('low', np.nan):,.0f})\n"
                    f"호크스값: {last_candle.get('v_hawk', np.nan):.4f} (5% 밴드: {last_candle.get('q05', np.nan):.4f}, 95% 밴드: {last_candle.get('q95', np.nan):.4f})\n"
                    f"현재 신호: {'매수' if last_candle.get('signal', 0) == 1 else '중립'}"
                )
                self.send_to_slack(candle_info)
            else:
                logging.warning("업데이트 후 trading_data가 비어있거나 필요한 열(signal, close 등)이 없습니다.")

            logging.info(f"데이터 업데이트 완료: {len(self.trading_data)}개 캔들")
            
        except Exception as e:
            error_msg = f"데이터 업데이트 중 심각한 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(f"스택 트레이스: {traceback.format_exc()}")
            self.send_to_slack(f"❌ {error_msg}")
            # 심각한 오류 시 이전 데이터라도 유지할지, 아니면 비울지 결정.
            # self.trading_data = pd.DataFrame() # 또는 이전 데이터 유지

    def check_signal(self):
        """현재 거래 신호 확인"""
        try:
            if self.trading_data.empty or 'signal' not in self.trading_data.columns:
                logging.warning("신호 확인을 위한 데이터가 충분하지 않습니다.")
                return None
            
            # 마지막 행의 signal 값을 가져옴. NaN일 경우 0으로 처리.
            current_signal = int(self.trading_data['signal'].iloc[-1]) if pd.notna(self.trading_data['signal'].iloc[-1]) else 0
            
            if current_signal != self.last_signal:
                signal_change_msg = f"신호 변경: {self.last_signal} -> {current_signal}"
                logging.info(signal_change_msg)
                
                signal_text_prev = "매수" if self.last_signal == 1 else "중립"
                signal_text_curr = "매수" if current_signal == 1 else "중립"
                self.send_to_slack(f"🔔 신호 변경: {signal_text_prev} -> {signal_text_curr}")
                
                self.last_signal = current_signal
                return current_signal
            
            return None # 신호 변경 없음
        except Exception as e:
            error_msg = f"신호 확인 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(f"스택 트레이스: {traceback.format_exc()}")
            self.send_to_slack(f"❌ {error_msg}")
            return None
    
    def execute_trade(self, signal):
        """거래 실행"""
        current_price = pyupbit.get_current_price(TICKER)
        if current_price is None:
            logging.error(f"{TICKER} 현재 가격 조회 실패. 거래를 실행할 수 없습니다.")
            self.send_to_slack(f"❌ {TICKER} 현재 가격 조회 실패로 거래를 실행할 수 없습니다.")
            return

        try:
            if self.trading_data.empty:
                logging.warning("거래 실행 위한 데이터 없음.")
                return

            latest_data = self.trading_data.iloc[-1]
            hawk_value = float(latest_data.get('v_hawk', np.nan)) # .get으로 안전하게 접근
            q95_value = float(latest_data.get('q95', np.nan))

            # 새로운 매수 신호
            if self.current_position == 0 and signal == 1:
                krw_balance = self.upbit.get_balance("KRW")
                if krw_balance is None: krw_balance = 0 # API 오류 시 대비
                
                if krw_balance > 5000: # 최소 주문 금액 (Upbit 기준 확인 필요)
                    buy_msg = f"매수 신호: {current_price:,.0f} KRW에 약 {krw_balance:,.0f} KRW 매수 시도"
                    logging.info(buy_msg)
                    self.send_to_slack(f"🔴 {buy_msg}")
                    
                    buy_amount_for_order = krw_balance * (1 - COMMISSION_RATE) # 수수료 선반영은 시장가에서 의미 없을 수 있음. Upbit은 주문 총액 기준.
                                                                                # 실제로는 krw_balance를 그대로 사용하거나, 아주 약간 낮은 금액 사용.
                    order = self.upbit.buy_market_order(TICKER, krw_balance * 0.999) # 전액 사용시 오류 가능성 줄이기 위해 약간 적게
                    
                    if order and isinstance(order, dict) and 'uuid' in order:
                        time.sleep(3) # 체결 대기 시간 증가
                        order_detail = self.upbit.get_order(order['uuid'])
                        
                        if order_detail and isinstance(order_detail, dict) and \
                           order_detail.get('state') == 'done' and order_detail.get('trades_count', 0) > 0:
                            
                            filled_volume = float(order_detail.get('executed_volume', 0))
                            avg_price = 0
                            if filled_volume > 0 and 'trades' in order_detail and order_detail['trades']:
                                total_value_executed = sum(float(t['price']) * float(t['volume']) for t in order_detail['trades'])
                                total_volume_executed = sum(float(t['volume']) for t in order_detail['trades'])
                                avg_price = total_value_executed / total_volume_executed if total_volume_executed > 0 else current_price
                            else: # trades 정보가 없거나 비정상적일 때
                                avg_price = current_price # 현재가로 대체 또는 주문 정보의 가격 사용
                                logging.warning("매수 주문 상세 정보에서 정확한 체결가 계산 불가. 현재가 또는 주문가 사용.")

                            self.current_position = 1
                            self.position_entry_price = avg_price
                            self.position_entry_time = datetime.datetime.now()
                            self.num_trades += 1
                            
                            trade_info = {
                                'time': self.position_entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'type': 'buy', 'price': avg_price, 'amount': filled_volume,
                                'value': avg_price * filled_volume, 'hawk_value': hawk_value, 'q95_value': q95_value
                            }
                            self.trade_history.append(trade_info)
                            
                            buy_result_msg = (
                                f"🔴 매수 체결 완료\n"
                                f"가격: {avg_price:,.0f} KRW\n"
                                f"수량: {filled_volume:.8f} BTC\n"
                                f"총액: {avg_price * filled_volume:,.0f} KRW\n"
                                f"호크스값: {hawk_value:.4f}, 95%밴드: {q95_value:.4f}\n"
                                f"총 거래: {self.num_trades}회"
                            )
                            logging.info(buy_result_msg.replace('\n', ', '))
                            self.send_to_slack(buy_result_msg)
                        else:
                            logging.warning(f"매수 주문 체결 실패 또는 부분 체결: {order_detail}")
                            self.send_to_slack(f"⚠️ 매수 주문 체결 확인 실패: {order.get('uuid', 'N/A')}")
                    else:
                        logging.error(f"매수 주문 실패: {order}")
                        self.send_to_slack(f"❌ 매수 주문 API 호출 실패. 응답: {str(order)[:100]}")
                else:
                    logging.info(f"매수 신호 무시: KRW 잔고 부족 ({krw_balance:,.0f} KRW)")
            
            # 매도 신호
            elif self.current_position == 1 and signal == 0:
                btc_balance = self.upbit.get_balance(TICKER.split('-')[1])
                if btc_balance is None: btc_balance = 0

                # BTC 최소 거래 가능 수량 확인 필요 (Upbit API 문서 참조)
                # 예: 0.00001 BTC 이상 등
                min_trade_btc = 0.00001 
                if btc_balance > min_trade_btc:
                    sell_msg = f"매도 신호: {current_price:,.0f} KRW에 {btc_balance:.8f} BTC 매도 시도"
                    logging.info(sell_msg)
                    self.send_to_slack(f"🔵 {sell_msg}")
                    
                    order = self.upbit.sell_market_order(TICKER, btc_balance)
                    
                    if order and isinstance(order, dict) and 'uuid' in order:
                        time.sleep(3)
                        order_detail = self.upbit.get_order(order['uuid'])

                        if order_detail and isinstance(order_detail, dict) and \
                           order_detail.get('state') == 'done' and order_detail.get('trades_count', 0) > 0:

                            filled_volume = float(order_detail.get('executed_volume', 0)) # 매도된 BTC 수량
                            avg_price = 0
                            if filled_volume > 0 and 'trades' in order_detail and order_detail['trades']:
                                total_value_executed = sum(float(t['price']) * float(t['volume']) for t in order_detail['trades'])
                                total_volume_executed = sum(float(t['volume']) for t in order_detail['trades']) # 이것이 filled_volume과 같아야 함
                                avg_price = total_value_executed / total_volume_executed if total_volume_executed > 0 else current_price
                            else:
                                avg_price = current_price
                                logging.warning("매도 주문 상세 정보에서 정확한 체결가 계산 불가. 현재가 또는 주문가 사용.")
                            
                            profit_pct = 0
                            if self.position_entry_price > 0: # 진입 가격이 있어야 수익률 계산 가능
                                profit_pct = (avg_price - self.position_entry_price) / self.position_entry_price
                            
                            self.current_position = 0
                            # self.num_trades += 1 # 이미 매수 시 카운트 했으므로, 왕복 거래를 1회로 센다면 여기서 증가시키지 않음. 편도 거래를 1회로 센다면 여기서도 증가. 현재는 편도.
                            
                            trade_info = {
                                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'type': 'sell', 'price': avg_price, 'amount': filled_volume,
                                'value': avg_price * filled_volume, 'profit_pct': profit_pct * 100,
                                'hawk_value': hawk_value
                            }
                            self.trade_history.append(trade_info)
                            
                            sell_result_msg = (
                                f"🔵 매도 체결 완료\n"
                                f"가격: {avg_price:,.0f} KRW\n"
                                f"수량: {filled_volume:.8f} BTC\n"
                                f"총액: {avg_price * filled_volume:,.0f} KRW\n"
                                f"수익률: {profit_pct*100:.2f}%\n"
                                f"호크스값: {hawk_value:.4f}\n"
                                f"총 거래: {self.num_trades}회"
                            )
                            logging.info(sell_result_msg.replace('\n', ', '))
                            self.send_to_slack(sell_result_msg)
                        else:
                            logging.warning(f"매도 주문 체결 실패 또는 부분 체결: {order_detail}")
                            self.send_to_slack(f"⚠️ 매도 주문 체결 확인 실패: {order.get('uuid', 'N/A')}")
                    else:
                        logging.error(f"매도 주문 실패: {order}")
                        self.send_to_slack(f"❌ 매도 주문 API 호출 실패. 응답: {str(order)[:100]}")
                else:
                    logging.info(f"매도 신호 무시: BTC 잔고 부족 또는 최소 거래 수량 미달 ({btc_balance:.8f} BTC)")

        except Exception as e:
            error_msg = f"거래 실행 중 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(f"스택 트레이스: {traceback.format_exc()}")
            self.send_to_slack(f"❌ {error_msg}")

    def log_account_info(self):
        """계좌 정보 로깅"""
        try:
            krw_balance = self.upbit.get_balance("KRW")
            btc_balance = self.upbit.get_balance("BTC") # TICKER의 base currency (BTC)
            
            if krw_balance is None: krw_balance = 0
            if btc_balance is None: btc_balance = 0

            current_btc_price = pyupbit.get_current_price(TICKER)
            if current_btc_price is None: current_btc_price = 0 # 가격 조회 실패 시
                
            btc_value_in_krw = btc_balance * current_btc_price
            total_value_in_krw = krw_balance + btc_value_in_krw
            
            position_status = "매수 중" if self.current_position == 1 else "중립"
            if self.current_position == 1 and self.position_entry_price > 0:
                position_status += f" (진입가: {self.position_entry_price:,.0f} KRW)"

            account_info_msg = (
                f"💰 계좌 정보\n"
                f"KRW 잔고: {krw_balance:,.2f} KRW\n"
                f"BTC 보유량: {btc_balance:.8f} BTC\n"
                f"BTC 평가액: {btc_value_in_krw:,.2f} KRW\n"
                f"총 자산 평가액: {total_value_in_krw:,.2f} KRW\n"
                f"현재 포지션: {position_status}\n"
                f"총 거래 횟수(편도): {self.num_trades}"
            )
            logging.info(account_info_msg.replace('\n',', '))
            self.send_to_slack(account_info_msg)
            
        except Exception as e:
            error_msg = f"계좌 정보 로깅 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            self.send_to_slack(f"❌ {error_msg}")

    def save_trade_history(self):
        if not self.trade_history:
            return
        try:
            # 날짜별 또는 월별로 파일 분리 저장 고려 가능
            history_filename = f'ec2_trade_history_{datetime.datetime.now().strftime("%Y%m%d")}.json'
            filepath = os.path.join(os.getcwd(), history_filename) # 로그 디렉토리 대신 현재 디렉토리
            
            # 기존 파일이 있으면 이어쓰기, 없으면 새로 만들기
            existing_history = []
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_history = json.load(f)
                    if not isinstance(existing_history, list): # 파일 내용이 리스트가 아니면 초기화
                        existing_history = []
                except json.JSONDecodeError:
                    logging.warning(f"{filepath} 파일이 JSON 형식이 아닙니다. 새로 덮어씁니다.")
                    existing_history = []
            
            # 현재 self.trade_history의 내용만 추가 (중복 방지 위해선 더 정교한 로직 필요)
            # 여기서는 간단히 현재 메모리의 trade_history를 기존 파일에 덮어쓰는 방식 대신,
            # 프로그램 실행 중 누적된 self.trade_history 전체를 저장
            # 또는, 새로운 거래만 추가하는 방식
            
            # 여기서는 self.trade_history가 실행 중 누적된 모든 거래를 담고 있다고 가정하고 전체 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.trade_history, f, indent=4) # indent로 가독성 높임
            
            logging.info(f"거래 기록 저장 완료: {filepath}")
        except Exception as e:
            error_msg = f"거래 기록 저장 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            self.send_to_slack(f"❌ {error_msg}")

    def calculate_performance_metrics(self):
        """백테스팅 성능 지표 계산 및 로깅 (self.trading_data 사용)"""
        if self.trading_data.empty or 'signal' not in self.trading_data.columns or 'close' not in self.trading_data.columns:
            logging.warning("성능 지표 계산을 위한 데이터 부족.")
            self.send_to_slack("⚠️ 성능 지표 계산 위한 데이터 부족.")
            return

        data = self.trading_data.copy() # 원본 데이터 변경 방지
        
        # 데이터가 충분한지 확인 (최소 LOOKBACK 기간 이상)
        if len(data) < VOLATILITY_LOOKBACK:
            logging.warning(f"성능 지표 계산 위한 데이터가 {VOLATILITY_LOOKBACK}개 미만입니다.")
            # self.send_to_slack(f"⚠️ 데이터 부족({len(data)}개)으로 성능 지표 계산 정확도 낮을 수 있음.")
            # return # 데이터 부족 시 계산하지 않거나, 경고만 하고 진행

        try:
            data['next_log_return'] = np.log(data['close'] / data['close'].shift(1)).shift(-1) # 다음 캔들의 로그 수익률
            data['signal_return'] = data['signal'].shift(1) * data['next_log_return'] # 현재 신호로 다음 캔들 수익률 얻음 (shift(1)로 현재 신호가 다음 캔들에 영향)
            data['signal_return'].fillna(0, inplace=True)

            win_returns = data[data['signal_return'] > 0]['signal_return'].sum()
            lose_returns = data[data['signal_return'] < 0]['signal_return'].abs().sum()
            
            signal_pf = win_returns / lose_returns if lose_returns > 0 else np.inf if win_returns > 0 else 0 # 손실 없으면 PF는 무한대 또는 0

            # 실제 거래 기반이 아닌, 신호 기반의 가상 거래 횟수
            signal_changes = data['signal'].diff().abs()
            # 0->1 (매수 진입), 1->0 (매수 청산)만 카운트 (왕복)
            # 또는 편도 거래 (0->1 또는 1->0) 각각을 카운트
            # 여기서는 신호가 0이 아닌 상태로 변경되는 것을 카운트 (0->1, 또는 0->-1 후 0으로 변경된 것)
            # 좀 더 명확하게는, 포지션 진입/청산 횟수를 세는 것이 좋음
            # get_trades_from_signal 함수를 활용하여 실제 거래 횟수 계산 가능
            
            # 여기서는 간단히 신호가 0이 아닌 기간의 비율
            time_in_market_pct = len(data[data['signal'] != 0]) / len(data) if len(data) > 0 else 0
            
            # get_trades_from_signal 사용하여 상세 분석 (DataFrame에 'signal' 열이 있어야 함)
            if 'signal' in data.columns:
                long_trades_df, _ = get_trades_from_signal(data, data['signal']) # 숏은 무시
                num_total_trades = len(long_trades_df)
                
                if num_total_trades > 0:
                    num_win_trades = len(long_trades_df[long_trades_df['percent'] > 0])
                    long_win_rate = num_win_trades / num_total_trades
                    avg_profit_per_trade = long_trades_df['percent'].mean() # 전체 거래 평균 수익률
                    avg_profit_on_wins = long_trades_df[long_trades_df['percent'] > 0]['percent'].mean()
                    avg_loss_on_losses = long_trades_df[long_trades_df['percent'] < 0]['percent'].mean()
                else:
                    long_win_rate = 0
                    avg_profit_per_trade = 0
                    avg_profit_on_wins = 0
                    avg_loss_on_losses = 0
            else:
                num_total_trades = 0
                long_win_rate = 0
                avg_profit_per_trade = 0
                # ... (다른 지표도 0으로)


            metrics_msg = (
                f"📊 백테스팅 성능 지표 (데이터 기반):\n"
                f"Profit Factor (로그수익률 기반): {signal_pf:.2f}\n"
                f"총 (가상)롱거래 횟수: {num_total_trades}\n"
                f"롱 승률: {long_win_rate:.2%}\n"
                f"롱 평균 수익/손실률: {avg_profit_per_trade:.2%}\n"
                f"  - 승리 시 평균: {avg_profit_on_wins:.2%}\n"
                f"  - 손실 시 평균: {avg_loss_on_losses:.2%}\n"
                f"시장 참여율 (신호!=0): {time_in_market_pct:.2%}"
            )
            
            logging.info(metrics_msg.replace('\n', ', '))
            self.send_to_slack(metrics_msg)
            
        except Exception as e:
            error_msg = f"성능 지표 계산 오류: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            self.send_to_slack(f"❌ {error_msg}")

    def calculate_backtest_metrics_for_chart(self, chart_df):
        """차트용 백테스트 성능 지표 계산 (입력된 DataFrame 사용)"""
        if chart_df.empty or 'signal' not in chart_df.columns or 'close' not in chart_df.columns:
            return {'profit_factor': 0, 'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0}
        
        data = chart_df.copy()
        try:
            # 간단한 누적 수익률 (로그 수익률 아님, 단순 수익률)
            # 신호가 1일때 다음날 시가 - 현재 종가 / 현재 종가
            # 또는 신호가 1일때 현재 종가 - 이전 종가 / 이전 종가 (이미 신호가 반영된 후)
            data['price_change_pct'] = data['close'].pct_change()
            data['signal_shifted'] = data['signal'].shift(1).fillna(0) # 어제 신호로 오늘 수익률을 먹는다
            data['strategy_return_pct'] = data['signal_shifted'] * data['price_change_pct']
            data['strategy_return_pct'].fillna(0, inplace=True)
            
            # 누적 수익률 (기하)
            data['cumulative_strategy_return'] = (1 + data['strategy_return_pct']).cumprod() -1
            total_return_pct = data['cumulative_strategy_return'].iloc[-1] * 100 if not data.empty else 0

            # 거래 기반 지표
            long_trades_df, _ = get_trades_from_signal(data, data['signal'])
            total_trades = len(long_trades_df)
            win_rate = 0
            if total_trades > 0:
                win_rate = (long_trades_df['percent'] > 0).sum() / total_trades * 100
            
            # Profit Factor (거래 기반)
            if not long_trades_df.empty:
                gross_profit = long_trades_df[long_trades_df['percent'] > 0]['percent'].sum() * 100 # 금액 기반이 더 정확하지만 여기선 %로
                gross_loss = abs(long_trades_df[long_trades_df['percent'] < 0]['percent'].sum()) * 100
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0
            else:
                profit_factor = 0

            return {
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return_pct': total_return_pct
            }
        except Exception as e:
            logging.error(f"차트용 백테스트 지표 계산 중 오류: {str(e)}")
            logging.error(traceback.format_exc())
            return {'profit_factor': 0, 'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0}

    def create_and_share_chart(self):
        """호크스 차트 생성 및 공유"""
        if self.trading_data.empty or len(self.trading_data) < VOLATILITY_LOOKBACK:
            logging.warning("차트 생성을 위한 데이터가 충분하지 않습니다.")
            # self.send_to_slack("⚠️ 차트 생성 위한 데이터 부족.") # 너무 잦은 알림 방지
            return None
        
        # 최근 500개 또는 전체 데이터 중 적은 것 사용
        chart_data_len = min(500, len(self.trading_data))
        chart_data = self.trading_data.iloc[-chart_data_len:].copy()

        if not all(col in chart_data.columns for col in ['v_hawk', 'q05', 'q95', 'signal', 'close']):
            logging.error("차트 생성에 필요한 열이 trading_data에 없습니다.")
            self.send_to_slack("❌ 차트 생성 실패: 필요한 데이터 열 누락.")
            return None
            
        try:
            # 매수/매도 포인트 (신호 변경 기준)
            buy_signals = chart_data[(chart_data['signal'] == 1) & (chart_data['signal'].shift(1) == 0)]
            sell_signals = chart_data[(chart_data['signal'] == 0) & (chart_data['signal'].shift(1) == 1)]

            backtest_metrics = self.calculate_backtest_metrics_for_chart(chart_data)
            
            chart_title = (
                f'Hawkes Process Trading ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) - {TICKER}<br>'
                f'KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK}<br>'
                f'<span style="color:cyan">PF: {backtest_metrics["profit_factor"]:.2f}</span>, '
                f'<span style="color:lightgreen">승률: {backtest_metrics["win_rate"]:.1f}%</span>, '
                f'<span style="color:orange">거래: {backtest_metrics["total_trades"]}회</span>, '
                f'<span style="color:magenta">누적수익: {backtest_metrics["total_return_pct"]:.2f}%</span> (차트 기간)'
            )
            
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                subplot_titles=('Price & Trades', 'Hawkes Volatility & Thresholds', 'Signal', 'Cumulative Return (%)'),
                row_heights=[0.4, 0.2, 0.15, 0.25]
            )

            # 1. 가격 및 거래
            fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data['open'], high=chart_data['high'], low=chart_data['low'], close=chart_data['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'] * 0.99, mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'] * 1.01, mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)

            # 2. 호크스 변동성
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['v_hawk'], name='V_Hawk', line=dict(color='yellow')), row=2, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['q05'], name='Q05', line=dict(color='lime', dash='dash')), row=2, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['q95'], name='Q95', line=dict(color='red', dash='dash')), row=2, col=1)

            # 3. 신호
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['signal'], name='Signal', line=dict(color='cyan', shape='hv')), row=3, col=1)
            fig.update_yaxes(tickvals=[0, 1], ticktext=['Neutral', 'Long'], range=[-0.1, 1.1], row=3, col=1)
            
            # 4. 누적 수익률 (차트 기간 기준)
            # calculate_backtest_metrics_for_chart에서 계산된 'cumulative_strategy_return' 사용
            chart_data['price_change_pct_vis'] = chart_data['close'].pct_change()
            chart_data['signal_shifted_vis'] = chart_data['signal'].shift(1).fillna(0)
            chart_data['strategy_return_pct_vis'] = chart_data['signal_shifted_vis'] * chart_data['price_change_pct_vis']
            chart_data['cumulative_strategy_return_vis'] = (1 + chart_data['strategy_return_pct_vis'].fillna(0)).cumprod() -1
            
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['cumulative_strategy_return_vis'] * 100, name='Cumulative Return', line=dict(color='magenta')), row=4, col=1)

            fig.update_layout(title_text=chart_title, template='plotly_dark', height=1000, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_xaxes(showticklabels=True) # 모든 x축 눈금 표시

            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hawkes_chart_{now_str}.html"
            filepath = os.path.join(self.charts_dir, filename)
            
            fig.write_html(filepath)
            logging.info(f"차트 저장됨: {filepath}")
            
            if self.charts_url_base:
                chart_url = f"{self.charts_url_base}/{filename}" # charts_dir가 웹 루트의 하위이므로 URL에 charts 포함
                self.send_to_slack(f"📊 호크스 분석 차트: <{chart_url}|차트 보기>")
                logging.info(f"차트 URL Slack 공유: {chart_url}")
            
            return filepath
        except Exception as e:
            logging.error(f"차트 생성 중 오류: {str(e)}")
            logging.error(traceback.format_exc())
            self.send_to_slack(f"❌ 차트 생성 실패: {str(e)}")
            return None
            
    def run(self):
        logging.info(f"EC2 Hawkes 트레이딩 봇 메인 루프 시작")
        
        # 초기 잔고 로깅 및 차트 생성은 load_initial_data에서 처리됨
        # self.log_account_info() # load_initial_data에서 호출
        if not self.trading_data.empty: # 초기 데이터 로드 성공 시에만 차트 생성
            self.create_and_share_chart()
            self.send_to_slack(f"📊 초기 호크스 차트 생성 완료 (데이터: {len(self.trading_data)}개).")
        else:
            self.send_to_slack(f"⚠️ 초기 데이터 로드 실패로 초기 차트 생성 건너뜀.")
            logging.error("초기 데이터 로드 실패로 프로그램 실행이 불안정할 수 있습니다. 종료합니다.")
            return # 심각한 오류로 간주하고 종료


        last_successful_update_time = datetime.datetime.now()
        last_chart_update_time = datetime.datetime.now()

        try:
            while True:
                loop_start_time = datetime.datetime.now()
                logging.info(f"메인 루프 사이클 시작: {loop_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                try:
                    self.update_data()
                    
                    if not self.trading_data.empty: # 데이터 업데이트 성공 시에만 진행
                        last_successful_update_time = datetime.datetime.now()
                        signal_change = self.check_signal()
                        
                        if signal_change is not None:
                            self.execute_trade(signal_change)
                            self.log_account_info() # 거래 후 계좌 정보 업데이트
                            self.create_and_share_chart() # 거래 후 차트 업데이트
                            last_chart_update_time = datetime.datetime.now()
                        
                        # 주기적 차트 업데이트 (예: 1시간마다)
                        if (datetime.datetime.now() - last_chart_update_time).total_seconds() >= 3600: # 1시간
                            if not self.trading_data.empty:
                                self.create_and_share_chart()
                                self.send_to_slack(f"📊 정기 호크스 차트 업데이트됨.")
                                last_chart_update_time = datetime.datetime.now()
                    else:
                        logging.warning("데이터 업데이트 후 trading_data가 비어있어 신호 확인 및 거래 건너뜀.")
                        # 데이터가 계속 비어있으면 문제 상황 알림
                        if (datetime.datetime.now() - last_successful_update_time).total_seconds() > 1800: # 30분 이상 업데이트 실패
                             self.send_to_slack("🚨 30분 이상 데이터 업데이트 실패 상태 지속 중. 확인 필요.")
                             logging.critical("30분 이상 데이터 업데이트 실패 상태 지속 중.")
                             # 여기서 프로그램을 안전하게 종료하거나, 재시작 로직 고려 가능

                    # 주기적 거래 기록 저장 (예: 10번 거래마다 또는 일정 시간마다)
                    if self.num_trades > 0 and self.num_trades % 5 == 0: # 5번의 편도 거래마다
                        self.save_trade_history()
                
                except Exception as inner_loop_e:
                    logging.error(f"메인 루프 내 개별 작업 오류: {str(inner_loop_e)}")
                    logging.error(traceback.format_exc())
                    self.send_to_slack(f"⚙️ 루프 내 작업 오류: {str(inner_loop_e)[:100]}")
                    # 오류 발생 시 짧은 대기 후 계속
                    time.sleep(60)


                # 다음 캔들 시간까지 대기 로직
                now = datetime.datetime.now()
                # 다음 정시 (+ 약간의 버퍼)
                next_run_time = (now + datetime.timedelta(hours=1)).replace(minute=0, second=15, microsecond=0)
                if next_run_time <= now : # 이미 다음 정시가 지났으면 그 다음 시간으로
                    next_run_time = (now + datetime.timedelta(hours=2)).replace(minute=0, second=15, microsecond=0)

                wait_seconds = (next_run_time - now).total_seconds()
                
                if wait_seconds <= 0: # 예상치 못한 경우, 짧게 대기
                    logging.warning(f"계산된 대기 시간이 0 또는 음수: {wait_seconds}초. 60초 대기합니다.")
                    wait_seconds = 60
                
                wait_msg = f"다음 사이클까지 {wait_seconds:.0f}초 대기 ({next_run_time.strftime('%Y-%m-%d %H:%M:%S')})"
                logging.info(wait_msg)
                # self.send_to_slack(f"⏳ {wait_msg}") # 너무 잦은 알림 방지

                # 대기 시간 동안 작은 단위로 나누어 슬립하면서 상태 확인 (예: 5분마다)
                heartbeat_interval = 1800 # 30분 (초)
                check_interval = 300    # 5분 (초)
                waited_time = 0
                last_heartbeat_time = now

                while waited_time < wait_seconds:
                    sleep_chunk = min(check_interval, wait_seconds - waited_time)
                    if sleep_chunk <=0: break # 이미 대기 시간 다 채웠으면 종료
                    time.sleep(sleep_chunk)
                    waited_time += sleep_chunk
                    
                    current_loop_time = datetime.datetime.now()
                    # 하트비트
                    if (current_loop_time - last_heartbeat_time).total_seconds() >= heartbeat_interval:
                        time_left = wait_seconds - waited_time
                        hb_msg = f"❤️ 아직 대기 중... 다음 사이클까지 약 {time_left/60:.0f}분 남음."
                        logging.info(hb_msg)
                        self.send_to_slack(hb_msg)
                        last_heartbeat_time = current_loop_time
                        # 간단한 API 연결 상태 확인 (선택적)
                        try:
                            api_check_price = pyupbit.get_current_price(TICKER)
                            if api_check_price is None:
                                logging.warning("하트비트 중 API 연결 확인 실패 (가격 조회 None).")
                        except Exception as api_e:
                            logging.warning(f"하트비트 중 API 연결 확인 오류: {str(api_e)}")
        
        except KeyboardInterrupt:
            stop_msg = "사용자에 의한 프로그램 종료 요청 (KeyboardInterrupt)"
            logging.info(stop_msg)
            self.send_to_slack(f"🛑 {stop_msg}")
        except Exception as e: # 메인 루프 자체의 심각한 오류
            error_msg = f"프로그램 실행 중 치명적 오류 발생: {str(e)}"
            logging.critical(error_msg)
            logging.critical(traceback.format_exc())
            self.send_to_slack(f"🚨 {error_msg}")
        finally:
            logging.info("프로그램 종료 절차 시작...")
            self.send_to_slack("🏁 프로그램 종료 절차 시작...")
            try:
                self.log_account_info()
            except Exception as final_log_e: logging.error(f"최종 계좌 로깅 실패: {final_log_e}")
            
            try:
                self.save_trade_history()
            except Exception as final_save_e: logging.error(f"최종 거래기록 저장 실패: {final_save_e}")
            
            try:
                if not self.trading_data.empty: self.create_and_share_chart()
            except Exception as final_chart_e: logging.error(f"최종 차트 생성 실패: {final_chart_e}")
            
            if self.httpd:
                try:
                    self.httpd.shutdown() # 웹서버 정상 종료
                    self.httpd.server_close()
                    logging.info("HTTP 서버 종료됨.")
                except Exception as http_shutdown_e:
                    logging.error(f"HTTP 서버 종료 중 오류: {http_shutdown_e}")

            end_msg = "프로그램이 종료되었습니다."
            logging.info(end_msg)
            self.send_to_slack(f"🏁 {end_msg}")

if __name__ == "__main__":
    if not UPBIT_ACCESS_KEY or not UPBIT_SECRET_KEY:
        print("치명적 오류: Upbit API 키가 .env 파일에 설정되지 않았습니다. 프로그램을 종료합니다.")
        logging.critical("Upbit API 키 미설정. 프로그램 종료.")
        exit(1)
    
    if not SLACK_API_TOKEN or not SLACK_CHANNEL:
        msg = "경고: Slack API 토큰 또는 채널이 .env 파일에 설정되지 않았습니다. Slack 알림이 비활성화됩니다."
        print(msg)
        logging.warning(msg)
    
    print(f"EC2 Hawkes 트레이딩 봇 인스턴스 생성 시도 (KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK})")
    logging.info(f"EC2 Hawkes 트레이딩 봇 인스턴스 생성 시도 (KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK})")
    
    try:
        trader = EC2HawkesTrader()
        trader.run()
    except Exception as main_exec_e:
        # EC2HawkesTrader 초기화 또는 run 시작 전 오류 처리
        print(f"봇 실행 중 최상위 레벨에서 오류 발생: {main_exec_e}")
        logging.critical(f"봇 실행 중 최상위 레벨에서 오류 발생: {main_exec_e}")
        logging.critical(traceback.format_exc())
        # Slack 알림 시도 (만약 slack_client가 초기화 되었다면)
        temp_slack_client = WebClient(token=SLACK_API_TOKEN) if SLACK_API_TOKEN and SLACK_CHANNEL else None
        if temp_slack_client:
            try:
                temp_slack_client.chat_postMessage(channel=SLACK_CHANNEL, text=f"🚨 봇 초기화 또는 실행 시작 단계에서 치명적 오류 발생하여 종료됨: {str(main_exec_e)[:200]}")
            except Exception as slack_send_fail_e:
                logging.error(f"최상위 오류 Slack 알림 실패: {slack_send_fail_e}")

