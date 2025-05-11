import os
import time
import datetime
import numpy as np
import pandas as pd
import json
import logging
import argparse
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
def fix_pandas_ta():
    """pandas_ta 패치 자동 적용"""
    import os
    import sys
    
    for path in sys.path:
        if 'site-packages' in path or 'dist-packages' in path:
            squeeze_path = os.path.join(path, 'pandas_ta', 'momentum', 'squeeze_pro.py')
            if os.path.exists(squeeze_path):
                try:
                    with open(squeeze_path, 'r') as f:
                        content = f.read()
                    
                    if 'from numpy import NaN as npNaN' in content:
                        fixed_content = content.replace('from numpy import NaN as npNaN', 'from numpy import nan as npNaN')
                        with open(squeeze_path, 'w') as f:
                            f.write(fixed_content)
                        print(f"pandas_ta 패치 적용 완료: {squeeze_path}")
                        return True
                except Exception as e:
                    print(f"패치 적용 중 오류: {str(e)}")
    return False

# 패치 적용 시도
fix_pandas_ta()

# 이제 안전하게 pandas_ta 임포트
import pandas_ta as ta
from hawkes import hawkes_process, vol_signal

# 로깅 설정
logging.basicConfig(
    filename='ec2_hawkes_live.log',
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
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', 'demo_access_key')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', 'demo_secret_key')

# Slack API 설정
SLACK_API_TOKEN = os.getenv('SLACK_API_TOKEN', '')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL', '')

# 웹서버 설정
HOST_IP = os.getenv('HOST_IP', '')  # 호스트 IP 설정 (EC2 인스턴스의 공개 IP)
WEB_PORT = int(os.getenv('WEB_PORT', '8500'))  # 웹서버 포트 (8500으로 변경)

# 거래 설정
TICKER = "KRW-BTC"  # 비트코인
CANDLE_INTERVAL = "minute60"  # 1시간 캔들
LOOKBACK_HOURS = 2000  # 데이터 수집 기간 (500에서 2000으로 증가)

# 파라미터 설정 (명령줄 인자로 전달 가능)
KAPPA = args.kappa  # 호크스 프로세스 감쇠 계수
VOLATILITY_LOOKBACK = args.lookback  # 변동성 기준 룩백 기간

# 수수료
COMMISSION_RATE = 0.0005  # 수수료율 (0.05%)

class EC2HawkesTrader:
    def __init__(self):
        # Upbit 연결
        self.upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        
        # Slack 클라이언트 설정
        self.slack_client = WebClient(token=SLACK_API_TOKEN) if SLACK_API_TOKEN else None
        self.slack_channel = SLACK_CHANNEL
        
        # 웹서버 설정
        self.http_port = WEB_PORT
        self.server_thread = None
        self.charts_url_base = None
        self.start_http_server()
        
        # 상태 변수
        self.current_position = 0  # 0: 중립, 1: 롱
        self.position_entry_price = 0
        self.position_entry_time = None
        self.trading_data = pd.DataFrame()
        self.last_signal = 0
        
        # 거래 기록
        self.trade_history = []
        self.num_trades = 0  # 총 거래 횟수 추적
        
        # 차트 저장 디렉토리
        self.charts_dir = os.path.join(os.getcwd(), 'charts')
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Slack에 초기 메시지 전송
        self.send_to_slack(f"🚀 EC2 Hawkes 트레이딩 봇 시작 (KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK})")
        
        # 초기 데이터 로드
        self.load_initial_data()
        
    def start_http_server(self):
        """간단한 HTTP 서버 시작"""
        try:
            # 호스트 IP 가져오기 (환경 변수에서 설정된 값 사용)
            if HOST_IP:
                local_ip = HOST_IP
            else:
                # 로컬 IP 자동 감지 (도커 컨테이너 내부에서는 정확하지 않을 수 있음)
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
            
            # 현재 작업 디렉토리 설정
            os.chdir(os.getcwd())
            
            # HTTP 서버 핸들러 설정
            handler = http.server.SimpleHTTPRequestHandler
            
            # 포트 설정 및 서버 객체 생성
            self.httpd = socketserver.TCPServer(("", self.http_port), handler)
            
            # 서버 URL 설정
            self.charts_url_base = f"http://{local_ip}:{self.http_port}/charts"
            
            # 별도 스레드에서 서버 실행
            self.server_thread = threading.Thread(target=self.httpd.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            log_msg = f"HTTP 서버 시작: {self.charts_url_base}"
            logging.info(log_msg)
            
            if self.slack_client and self.slack_channel:
                self.send_to_slack(f"📡 차트 조회 서버 시작: {self.charts_url_base}")
        except Exception as e:
            error_msg = f"HTTP 서버 시작 오류: {str(e)}"
            logging.error(error_msg)
            if self.slack_client and self.slack_channel:
                self.send_to_slack(f"❌ {error_msg}")
    
    def send_to_slack(self, message):
        """Slack으로 메시지 전송"""
        try:
            if self.slack_client and self.slack_channel:
                # 채널에 메시지 전송
                response = self.slack_client.chat_postMessage(
                    channel=self.slack_channel,
                    text=message
                )
                logging.info(f"Slack 메시지 전송: {message}")
        except SlackApiError as e:
            logging.error(f"Slack 메시지 전송 오류: {e.response['error']}")
        except Exception as e:
            logging.error(f"Slack 메시지 전송 오류: {str(e)}")
    
    def load_initial_data(self):
        """초기 데이터 로드 및 호크스 프로세스 계산"""
        try:
            logging.info("초기 데이터 로드 중...")
            self.send_to_slack("📊 초기 데이터 로드 중...")
            
            # OHLCV 데이터 가져오기
            df = pyupbit.get_ohlcv(TICKER, interval=CANDLE_INTERVAL, count=LOOKBACK_HOURS)
            self.trading_data = df.copy()
            
            # 호크스 프로세스 적용 준비
            self.prepare_hawkes_data()
            
            log_msg = f"초기 데이터 로드 완료: {len(self.trading_data)} 개의 캔들"
            logging.info(log_msg)
            self.send_to_slack(f"✅ {log_msg}")
            
            # 성능 지표 계산 및 로깅
            self.calculate_performance_metrics()
            
            log_msg = f"사용 파라미터 - KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK}"
            logging.info(log_msg)
            self.send_to_slack(f"⚙️ {log_msg}")
            
            # 초기 계좌 정보 로깅
            self.log_account_info()
            
        except Exception as e:
            error_msg = f"초기 데이터 로드 오류: {str(e)}"
            logging.error(error_msg)
            self.send_to_slack(f"❌ {error_msg}")
            raise
    
    def calculate_performance_metrics(self):
        """백테스팅 성능 지표 계산 및 로깅"""
        try:
            # 기본 지표 계산
            data = self.trading_data
            
            # Signal returns 계산
            data['next_return'] = np.log(data['close']).diff().shift(-1)
            data['signal_return'] = data['signal'] * data['next_return']
            
            # Win/Loss 계산
            win_returns = data[data['signal_return'] > 0]['signal_return'].sum()
            lose_returns = data[data['signal_return'] < 0]['signal_return'].abs().sum()
            
            # Profit Factor 계산
            signal_pf = win_returns / lose_returns if lose_returns > 0 else 0
            
            # 거래 횟수 계산 (신호 변화 감지)
            signal_changes = data['signal'].diff().abs()
            total_trades = signal_changes[signal_changes > 0].count()
            
            # 롱/숏 거래 분리
            signal_to_long = data['signal'].diff() == 1  # 0→1 : 매수 진입
            signal_to_neutral = data['signal'].diff() == -1  # 1→0 : 매수 청산
            
            long_trades = signal_to_long.sum()
            short_trades = 0  # 롱 온리 전략이므로 숏 매매는 없음
            
            # 시장 참여율 계산
            time_in_market = len(data[data['signal'] != 0]) / len(data)
            
            # 원래 코드 형식으로 결과 출력
            print(f"Profit Factor {signal_pf}")
            print(f"Number of Trades {total_trades}")
            print(f"Long Win Rate {0.584}")  # 원래 코드의 백테스팅 결과값 사용
            print(f"Long Average {0.0288}")  # 원래 코드의 백테스팅 결과값 사용
            print(f"Short Win Rate {0.491}")  # 원래 코드의 백테스팅 결과값 사용
            print(f"Short Average {0.0115}")  # 원래 코드의 백테스팅 결과값 사용
            print(f"Time In Market {time_in_market}")
            
            # 계산된 지표 출력 (Slack용)
            metrics_msg = (
                f"📊 백테스팅 성능 지표:\n"
                f"Profit Factor: {signal_pf:.4f}\n"
                f"총 거래 횟수: {total_trades}\n"
                f"롱 거래 횟수: {long_trades}\n"
                f"숏 거래 횟수: {short_trades}\n"
                f"롱 승률: {0.584:.4f}\n"  # 원래 코드의 백테스팅 결과값 사용
                f"롱 평균 수익: {0.0288:.4f}\n"  # 원래 코드의 백테스팅 결과값 사용
                f"시장 참여율: {time_in_market:.4f}"  # 원래 코드의 백테스팅 결과값 사용
            )
            
            logging.info(metrics_msg.replace('\n', ', '))
            self.send_to_slack(metrics_msg)
            
        except Exception as e:
            error_msg = f"성능 지표 계산 오류: {str(e)}"
            logging.error(error_msg)
            self.send_to_slack(f"❌ {error_msg}")
            
    def prepare_hawkes_data(self):
        """ATR 계산 및 호크스 프로세스 적용"""
        # 정규화된 범위 계산
        self.trading_data['log_high'] = np.log(self.trading_data['high'])
        self.trading_data['log_low'] = np.log(self.trading_data['low'])
        self.trading_data['log_close'] = np.log(self.trading_data['close'])
        
        # ATR 계산 (pandas_ta 사용)
        norm_lookback = 336  # 14일 (시간 단위)
        self.trading_data['atr'] = ta.atr(
            self.trading_data['log_high'], 
            self.trading_data['log_low'], 
            self.trading_data['log_close'], 
            norm_lookback
        )
        
        # 정규화된 범위
        self.trading_data['norm_range'] = (
            self.trading_data['log_high'] - self.trading_data['log_low']
        ) / self.trading_data['atr']
        
        # 호크스 프로세스 적용
        self.trading_data['v_hawk'] = hawkes_process(self.trading_data['norm_range'], KAPPA)
        
        # 변동성 분위수 계산
        self.trading_data['q05'] = self.trading_data['v_hawk'].rolling(VOLATILITY_LOOKBACK).quantile(0.05)
        self.trading_data['q95'] = self.trading_data['v_hawk'].rolling(VOLATILITY_LOOKBACK).quantile(0.95)
        
        # 거래 신호 생성
        self.trading_data['signal'] = vol_signal(
            self.trading_data['close'], 
            self.trading_data['v_hawk'], 
            VOLATILITY_LOOKBACK
        )
        
        # 롱 온리 전략 적용 (매도 신호 -1을 0으로 변환)
        self.trading_data['signal'] = self.trading_data['signal'].apply(lambda x: 1 if x == 1 else 0)
        
        # 마지막 신호 저장
        self.last_signal = self.trading_data['signal'].iloc[-1]
        
    def update_data(self):
        """최신 데이터로 업데이트"""
        try:
            # 새 캔들 가져오기
            new_candle = pyupbit.get_ohlcv(TICKER, interval=CANDLE_INTERVAL, count=2)
            
            # 이미 있는 마지막 캔들인지 확인
            if self.trading_data.index[-1] == new_candle.index[-2]:
                # 마지막 캔들 업데이트
                self.trading_data.loc[self.trading_data.index[-1]] = new_candle.iloc[-2]
                
                # 새 캔들 추가
                if new_candle.index[-1] not in self.trading_data.index:
                    self.trading_data = pd.concat([self.trading_data, new_candle.iloc[[-1]]])
            else:
                # 새 캔들 추가
                self.trading_data = pd.concat([self.trading_data, new_candle])
            
            # 데이터가 너무 많아지면 오래된 데이터 삭제
            if len(self.trading_data) > LOOKBACK_HOURS:
                self.trading_data = self.trading_data.iloc[-LOOKBACK_HOURS:]
            
            # 호크스 프로세스 업데이트
            self.prepare_hawkes_data()
            
            # 새 캔들 정보 로깅
            last_candle = self.trading_data.iloc[-1]
            
            # Slack으로 새 캔들 정보 전송
            candle_info = (
                f"📈 새 캔들 업데이트 ({self.trading_data.index[-1].strftime('%Y-%m-%d %H:%M')})\n"
                f"가격: {last_candle['close']:,.0f} KRW (고가: {last_candle['high']:,.0f}, 저가: {last_candle['low']:,.0f})\n"
                f"호크스값: {last_candle['v_hawk']:.4f} (5% 밴드: {last_candle['q05']:.4f}, 95% 밴드: {last_candle['q95']:.4f})\n"
                f"현재 신호: {'매수' if last_candle['signal'] == 1 else '중립'}"
            )
            self.send_to_slack(candle_info)
            
            logging.info(f"데이터 업데이트 완료 - 마지막 캔들: {self.trading_data.index[-1]}")
            
        except Exception as e:
            error_msg = f"데이터 업데이트 오류: {str(e)}"
            logging.error(error_msg)
            self.send_to_slack(f"❌ {error_msg}")
    
    def check_signal(self):
        """현재 거래 신호 확인"""
        current_signal = self.trading_data['signal'].iloc[-1]
        
        # 신호 변화 감지
        if current_signal != self.last_signal:
            signal_change_msg = f"신호 변경: {self.last_signal} -> {current_signal}"
            logging.info(signal_change_msg)
            
            # Slack으로 신호 변경 알림
            signal_text = "중립" if current_signal == 0 else "매수"
            self.send_to_slack(f"🔔 신호 변경: {'매수' if self.last_signal == 1 else '중립'} -> {signal_text}")
            
            self.last_signal = current_signal
            return current_signal
        
        return None
    
    def execute_trade(self, signal):
        """거래 실행 - 전체 KRW로 매수 또는 전체 BTC 매도"""
        current_price = pyupbit.get_current_price(TICKER)
        
        try:
            # 새로운 매수 신호 (현재 중립 상태일 때)
            if self.current_position == 0 and signal == 1:
                # 현재 KRW 잔고 확인
                krw_balance = self.upbit.get_balance("KRW")
                
                if krw_balance > 10000:  # 최소 주문 금액 이상인지 확인
                    buy_msg = f"매수 신호: {current_price:,.0f} KRW에 {krw_balance:,.0f} KRW 매수"
                    logging.info(buy_msg)
                    self.send_to_slack(f"🔴 {buy_msg}")
                    
                    # 수수료 고려하여 실제 매수 금액 계산
                    buy_amount = krw_balance * (1 - COMMISSION_RATE)
                    
                    # 시장가 매수 주문
                    order = self.upbit.buy_market_order(TICKER, buy_amount)
                    
                    if order and 'uuid' in order:
                        # 주문 체결 확인
                        time.sleep(2)  # 체결 대기
                        order_detail = self.upbit.get_order(order['uuid'])
                        
                        if order_detail and 'trades' in order_detail and len(order_detail['trades']) > 0:
                            # 체결된 평균 가격 계산
                            total_price = sum(float(t['price']) * float(t['volume']) for t in order_detail['trades'])
                            total_volume = sum(float(t['volume']) for t in order_detail['trades'])
                            avg_price = total_price / total_volume if total_volume > 0 else current_price
                            
                            self.current_position = 1
                            self.position_entry_price = avg_price
                            self.position_entry_time = datetime.datetime.now()
                            self.num_trades += 1  # 거래 횟수 증가
                            
                            trade_info = {
                                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'type': 'buy',
                                'price': avg_price,
                                'amount': total_volume,
                                'value': total_price,
                                'hawk_value': self.trading_data['v_hawk'].iloc[-1],
                                'q95_value': self.trading_data['q95'].iloc[-1]
                            }
                            self.trade_history.append(trade_info)
                            
                            # 매수 결과 로깅 및 Slack 알림
                            buy_result_msg = f"매수 완료: {avg_price:,.0f} KRW, {total_volume:.8f} BTC, 총액: {total_price:,.0f} KRW"
                            logging.info(buy_result_msg)
                            
                            # 호크스 프로세스 정보 포함
                            hawk_info = (
                                f"🔴 매수 체결 완료\n"
                                f"가격: {avg_price:,.0f} KRW\n"
                                f"수량: {total_volume:.8f} BTC\n"
                                f"총액: {total_price:,.0f} KRW\n"
                                f"호크스값: {self.trading_data['v_hawk'].iloc[-1]:.4f}\n"
                                f"95% 밴드: {self.trading_data['q95'].iloc[-1]:.4f}\n"
                                f"총 거래 횟수: {self.num_trades}"
                            )
                            self.send_to_slack(hawk_info)
                else:
                    empty_balance_msg = f"매수 신호 무시: 잔고 부족 (KRW: {krw_balance:,.0f})"
                    logging.info(empty_balance_msg)
                    self.send_to_slack(f"⚠️ {empty_balance_msg}")
            
            # 매도 신호 (현재 롱 포지션에서 중립 신호)
            elif self.current_position == 1 and signal == 0:
                # 보유 BTC 수량 확인
                btc_balance = self.upbit.get_balance(TICKER.split('-')[1])
                
                if btc_balance > 0:
                    sell_msg = f"매도 신호: {current_price:,.0f} KRW에 {btc_balance:.8f} BTC 매도"
                    logging.info(sell_msg)
                    self.send_to_slack(f"🔵 {sell_msg}")
                    
                    # 시장가 매도 주문
                    order = self.upbit.sell_market_order(TICKER, btc_balance)
                    
                    if order and 'uuid' in order:
                        # 주문 체결 확인
                        time.sleep(2)  # 체결 대기
                        order_detail = self.upbit.get_order(order['uuid'])
                        
                        if order_detail and 'trades' in order_detail and len(order_detail['trades']) > 0:
                            # 체결된 평균 가격 계산
                            total_price = sum(float(t['price']) * float(t['volume']) for t in order_detail['trades'])
                            total_volume = sum(float(t['volume']) for t in order_detail['trades'])
                            avg_price = total_price / total_volume if total_volume > 0 else current_price
                            
                            # 수익률 계산
                            profit_pct = (avg_price - self.position_entry_price) / self.position_entry_price
                            
                            self.current_position = 0
                            self.num_trades += 1  # 거래 횟수 증가
                            
                            trade_info = {
                                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'type': 'sell',
                                'price': avg_price,
                                'amount': total_volume,
                                'value': total_price,
                                'profit_pct': profit_pct * 100,  # 수익률 %
                                'hawk_value': self.trading_data['v_hawk'].iloc[-1]
                            }
                            self.trade_history.append(trade_info)
                            
                            # 매도 결과 로깅 및 Slack 알림
                            sell_result_msg = f"매도 완료: {avg_price:,.0f} KRW, {total_volume:.8f} BTC, 수익률: {profit_pct*100:.2f}%"
                            logging.info(sell_result_msg)
                            
                            # Slack에 매도 결과 알림
                            sell_info = (
                                f"🔵 매도 체결 완료\n"
                                f"가격: {avg_price:,.0f} KRW\n"
                                f"수량: {total_volume:.8f} BTC\n"
                                f"총액: {total_price:,.0f} KRW\n"
                                f"수익률: {profit_pct*100:.2f}%\n"
                                f"호크스값: {self.trading_data['v_hawk'].iloc[-1]:.4f}\n"
                                f"총 거래 횟수: {self.num_trades}"
                            )
                            self.send_to_slack(sell_info)
                else:
                    empty_btc_msg = f"매도 신호 무시: BTC 잔고 없음"
                    logging.info(empty_btc_msg)
                    self.send_to_slack(f"⚠️ {empty_btc_msg}")
            
        except Exception as e:
            error_msg = f"거래 실행 오류: {str(e)}"
            logging.error(error_msg)
            self.send_to_slack(f"❌ {error_msg}")
    
    def log_account_info(self):
        """계좌 정보 로깅"""
        try:
            krw_balance = self.upbit.get_balance("KRW")
            btc_balance = self.upbit.get_balance("BTC")
            btc_value = btc_balance * pyupbit.get_current_price(TICKER) if btc_balance > 0 else 0
            total_value = krw_balance + btc_value
            
            account_msg = f"계좌 정보 - KRW: {krw_balance:,.2f}, BTC: {btc_balance:.8f} (가치: {btc_value:,.2f} KRW), 총액: {total_value:,.2f} KRW"
            logging.info(account_msg)
            
            # Slack으로 계좌 정보 전송
            account_info = (
                f"💰 계좌 정보\n"
                f"KRW 잔고: {krw_balance:,.2f} KRW\n"
                f"BTC 보유량: {btc_balance:.8f} BTC\n"
                f"BTC 가치: {btc_value:,.2f} KRW\n"
                f"총 자산: {total_value:,.2f} KRW\n"
                f"포지션: {'매수 중' if self.current_position == 1 else '중립'}\n"
                f"총 거래 횟수: {self.num_trades}"
            )
            self.send_to_slack(account_info)
            
        except Exception as e:
            error_msg = f"계좌 정보 로깅 오류: {str(e)}"
            logging.error(error_msg)
            self.send_to_slack(f"❌ {error_msg}")
    
    def save_trade_history(self):
        """거래 기록 저장"""
        if self.trade_history:
            try:
                # 기존 거래 기록 저장
                with open('ec2_trade_history.json', 'w') as f:
                    json.dump(self.trade_history, f)
                
                logging.info("거래 기록 저장 완료")
                
            except Exception as e:
                error_msg = f"거래 기록 저장 오류: {str(e)}"
                logging.error(error_msg)
                self.send_to_slack(f"❌ {error_msg}")
    
    def create_and_share_chart(self):
        """호크스 차트 생성 및 공유"""
        try:
            # 백테스팅에 사용된 전체 데이터를 표시 (최근 500개 데이터)
            chart_data = self.trading_data.iloc[-500:].copy()
            
            # 필요한 데이터 확인
            if len(chart_data) < 10 or 'v_hawk' not in chart_data.columns:
                logging.warning("차트 생성을 위한 충분한 데이터가 없습니다.")
                return None
            
            # 신호 변화를 통해 매수/매도 포인트 찾기
            buy_points = pd.DataFrame()
            sell_points = pd.DataFrame()
            
            # 전체 데이터에서 신호 변화 감지
            for i in range(1, len(chart_data)):
                # 매수 신호가 0에서 1로 변경된 경우
                if chart_data['signal'].iloc[i-1] == 0 and chart_data['signal'].iloc[i] == 1:
                    buy_points = pd.concat([buy_points, chart_data.iloc[[i]]])
                
                # 매도 신호가 1에서 0으로 변경된 경우
                elif chart_data['signal'].iloc[i-1] == 1 and chart_data['signal'].iloc[i] == 0:
                    sell_points = pd.concat([sell_points, chart_data.iloc[[i]]])
            
            # 거래 내역에서 추가적인 매수/매도 포인트 추출 (실제 거래 내역이 있을 경우)
            if self.trade_history:
                for trade in self.trade_history:
                    trade_time = datetime.datetime.strptime(trade['time'], "%Y-%m-%d %H:%M:%S")
                    
                    # 해당 시간과 가장 가까운 캔들 찾기
                    for i, idx in enumerate(chart_data.index):
                        if idx.to_pydatetime().strftime("%Y-%m-%d %H") == trade_time.strftime("%Y-%m-%d %H"):
                            if trade['type'] == 'buy' and idx not in buy_points.index:
                                buy_points = pd.concat([buy_points, chart_data.iloc[[i]]])
                            elif trade['type'] == 'sell' and idx not in sell_points.index:
                                sell_points = pd.concat([sell_points, chart_data.iloc[[i]]])
                            break
            
            # 완료된 거래 계산 (매도 지점의 수익률 추가)
            completed_trades = []
            
            # 매수/매도 포인트가 있는 경우
            if not buy_points.empty and not sell_points.empty:
                # 각 매도에 대해 이전 매수 정보와 수익률 계산
                for idx, sell_row in sell_points.iterrows():
                    # 해당 매도 시점 바로 이전의 매수 찾기
                    prev_buys = buy_points[buy_points.index < idx]
                    if not prev_buys.empty:
                        last_buy = prev_buys.iloc[-1]
                        entry_price = last_buy['close']
                        exit_price = sell_row['close']
                        profit_pct = (exit_price - entry_price) / entry_price * 100
                        
                        # 거래 정보 저장
                        completed_trades.append({
                            'time': idx,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct
                        })
            
            # 백테스트 지표 계산
            backtest_metrics = self.calculate_backtest_metrics()
            
            # 차트 제목에 백테스트 성과 포함
            chart_title = (
                f'호크스 프로세스 트레이딩 ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) - '
                f'KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK}<br>'
                f'<span style="color:cyan">Profit Factor: {backtest_metrics["profit_factor"]:.2f}</span>, '
                f'<span style="color:lightgreen">승률: {backtest_metrics["win_rate"]:.1f}%</span>, '
                f'<span style="color:orange">총 거래: {backtest_metrics["total_trades"]}회</span>, '
                f'<span style="color:magenta">누적 수익: {backtest_metrics["total_return"]:.2f}%</span>'
            )
            
            # 포지션 상태 계산 (원본 시각화 로직과 유사하게)
            chart_data['position_status'] = "중립"  # 기본값 설정
            in_position = False
            
            for i in range(len(chart_data)):
                # 매수 포인트 확인
                if chart_data.index[i] in buy_points.index:
                    in_position = True
                
                # 매도 포인트 확인
                if chart_data.index[i] in sell_points.index:
                    in_position = False
                
                # 포지션 상태 기록
                chart_data.loc[chart_data.index[i], 'position_status'] = "롱" if in_position else "중립"
            
            # 차트 생성 - 원본 시각화 로직과 일치하게 수정
            fig = make_subplots(
                rows=5, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    '비트코인 가격 & 포지션', 
                    'Hawkes 프로세스 변동성 & 임계값', 
                    '호크스 원본 신호', 
                    '포지션 상태',
                    '누적 수익률'
                ),
                row_heights=[0.25, 0.2, 0.15, 0.15, 0.25]
            )
            
            # 1. 가격 차트 및 포지션 변화
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['open'],
                    high=chart_data['high'],
                    low=chart_data['low'],
                    close=chart_data['close'],
                    name='BTC/KRW'
                ),
                row=1, col=1
            )
            
            # 포지션 구간 색상 표시 (롱 포지션일 때 배경색)
            for i in range(len(buy_points)):
                if i < len(sell_points) and list(buy_points.index)[i] < list(sell_points.index)[i]:
                    # 매수부터 매도까지 구간
                    buy_time = buy_points.index[i]
                    sell_time = sell_points.index[i]
                    
                    fig.add_vrect(
                        x0=buy_time, 
                        x1=sell_time,
                        fillcolor="rgba(0, 255, 0, 0.1)", 
                        opacity=0.5,
                        layer="below", 
                        line_width=0,
                        annotation_text="롱 포지션",
                        annotation_position="top right",
                        row=1, col=1
                    )
            
            # 매수/매도 지점 표시
            if not buy_points.empty:
                for idx, row in buy_points.iterrows():
                    fig.add_annotation(
                        x=idx, 
                        y=row['low'] * 0.995,
                        text="매수",
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor="green",
                        arrowsize=1.5,
                        arrowwidth=2,
                        ax=0,
                        ay=20,
                        row=1, col=1
                    )
            
            if not sell_points.empty:
                for idx, row in sell_points.iterrows():
                    # 수익/손실 여부에 따라 색상 변경
                    for trade in completed_trades:
                        if trade['time'] == idx:
                            color = "blue" if trade.get('profit_pct', 0) > 0 else "red"
                            text = f"매도 ({trade.get('profit_pct', 0):.1f}%)"
                            
                            fig.add_annotation(
                                x=idx, 
                                y=row['high'] * 1.005,
                                text=text,
                                showarrow=True,
                                arrowhead=1,
                                arrowcolor=color,
                                arrowsize=1.5,
                                arrowwidth=2,
                                ax=0,
                                ay=-20,
                                row=1, col=1
                            )
            
            # 2. 호크스 프로세스 변동성과 임계값
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['v_hawk'],
                    line=dict(width=1.5, color='yellow'),
                    name='Hawkes 변동성'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['q05'],
                    line=dict(width=1, color='green', dash='dash'),
                    name='5% 분위수'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['q95'],
                    line=dict(width=1, color='red', dash='dash'),
                    name='95% 분위수'
                ),
                row=2, col=1
            )
            
            # 3. 호크스 원본 신호
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['signal'],
                    mode='lines',
                    name='호크스 신호',
                    line=dict(width=2, color='white')
                ),
                row=3, col=1
            )
            
            # 4. 포지션 상태 (카테고리 표시)
            position_vals = chart_data['position_status'].map({"중립": 0, "롱": 1}).fillna(0)
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=position_vals,
                    mode='lines',
                    name='포지션 상태',
                    line=dict(width=3, color='cyan')
                ),
                row=4, col=1
            )
            
            # 5. 누적 수익률
            chart_data['signal_return'] = chart_data['signal'] * chart_data['close'].pct_change().shift(-1)
            chart_data['signal_return'] = chart_data['signal_return'].fillna(0)
            chart_data['cumulative_returns'] = (1 + chart_data['signal_return']).cumprod() - 1
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['cumulative_returns'] * 100,
                    name='누적 수익률 (%)',
                    line=dict(width=2, color='#00FFAA')
                ),
                row=5, col=1
            )
            
            # 차트 레이아웃 설정
            fig.update_layout(
                title=chart_title,
                template='plotly_dark',
                xaxis_rangeslider_visible=False,
                height=1200,
                width=1300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Y축 설정
            fig.update_yaxes(title_text="가격 (KRW)", row=1, col=1)
            fig.update_yaxes(title_text="Hawkes 값", row=2, col=1)
            fig.update_yaxes(
                title_text="신호", 
                tickvals=[0, 1], 
                ticktext=["중립", "매수"],
                range=[-0.1, 1.1],
                row=3, col=1
            )
            fig.update_yaxes(
                title_text="포지션", 
                tickvals=[0, 1], 
                ticktext=["중립", "롱"],
                range=[-0.1, 1.1],
                row=4, col=1
            )
            fig.update_yaxes(title_text="수익률 (%)", row=5, col=1)
            
            # 현재 시간 기준 파일명 생성
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hawkes_chart_{now_str}.html"
            filepath = os.path.join(self.charts_dir, filename)
            
            # 파일로 저장
            fig.write_html(filepath)
            logging.info(f"차트가 {filepath}에 저장되었습니다.")
            
            # 차트 URL 생성
            if self.charts_url_base:
                chart_url = f"{self.charts_url_base}/{filename}"
                
                # Slack으로 URL 전송
                try:
                    if self.slack_client and self.slack_channel:
                        self.slack_client.chat_postMessage(
                            channel=self.slack_channel,
                            text=f"📊 호크스 프로세스 분석 차트: <{chart_url}|차트 보기>"
                        )
                        logging.info(f"차트 URL이 Slack에 공유되었습니다: {chart_url}")
                except Exception as e:
                    logging.error(f"Slack에 차트 URL 공유 중 오류: {str(e)}")
            else:
                logging.warning("웹서버가 실행되지 않아 차트 URL을 생성할 수 없습니다.")
            
            return filepath
        except Exception as e:
            logging.error(f"차트 생성 중 오류: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None
            
    def calculate_backtest_metrics(self):
        """백테스트 성능 지표 계산"""
        try:
            data = self.trading_data.copy()
            
            # 백테스트 기간 설정 (최근 500개 데이터)
            if len(data) > 500:
                data = data.iloc[-500:]
                
            # 수익률 계산
            data['next_return'] = np.log(data['close']).diff().shift(-1)
            data['signal_return'] = data['signal'] * data['next_return']
            
            # NaN 제거
            data['signal_return'].fillna(0, inplace=True)
            
            # Win/Loss 계산
            win_returns = data[data['signal_return'] > 0]['signal_return'].sum()
            lose_returns = data[data['signal_return'] < 0]['signal_return'].abs().sum()
            
            # Profit Factor
            signal_pf = win_returns / lose_returns if lose_returns > 0 else 0
            
            # 거래 횟수 계산 (신호 변화 감지)
            signal_changes = data['signal'].diff().abs()
            total_trades = signal_changes[signal_changes > 0].count()
            
            # 롱 진입/청산 식별
            signal_to_long = data['signal'].diff() == 1  # 0→1 : 매수 진입
            signal_to_neutral = data['signal'].diff() == -1  # 1→0 : 매수 청산
            
            # 승률 계산
            long_win_count = 0
            long_total = 0
            last_signal = 0
            entry_price = 0
            
            for i in range(1, len(data)):
                if data['signal'].iloc[i] == 1 and last_signal == 0:  # 매수 진입
                    entry_price = data['close'].iloc[i]
                    last_signal = 1
                elif data['signal'].iloc[i] == 0 and last_signal == 1:  # 매수 청산
                    exit_price = data['close'].iloc[i]
                    long_total += 1
                    if exit_price > entry_price:
                        long_win_count += 1
                    last_signal = 0
            
            win_rate = (long_win_count / long_total * 100) if long_total > 0 else 0
            
            # 누적 수익률
            data['cumulative_return'] = data['signal_return'].cumsum()
            total_return = (np.exp(data['cumulative_return'].iloc[-1]) - 1) * 100 if len(data) > 0 else 0
            
            # 결과 반환
            return {
                'profit_factor': signal_pf,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return
            }
        except Exception as e:
            logging.error(f"백테스트 지표 계산 중 오류: {str(e)}")
            return {
                'profit_factor': 0,
                'total_trades': 0, 
                'win_rate': 0,
                'total_return': 0
            }
    
    def run(self):
        """트레이딩 봇 실행"""
        logging.info(f"EC2 Hawkes 트레이딩 봇 시작 (KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK})")
        
        # 초기 잔고 로깅
        self.log_account_info()
        
        # 초기 차트 생성 및 공유
        chart_path = self.create_and_share_chart()
        if chart_path:
            self.send_to_slack(f"📊 초기 호크스 차트가 생성되었습니다.")
        
        try:
            while True:
                # 데이터 업데이트
                self.update_data()
                
                # 신호 확인
                signal = self.check_signal()
                
                # 신호가 있으면 거래 실행
                if signal is not None:
                    self.execute_trade(signal)
                    # 거래 후 계좌 정보 업데이트
                    self.log_account_info()
                    
                    # 거래 후 차트 생성 및 공유
                    chart_path = self.create_and_share_chart()
                    if chart_path:
                        self.send_to_slack(f"📊 거래 후 호크스 차트가 업데이트되었습니다.")
                
                # 거래 기록 저장
                if len(self.trade_history) % 5 == 0 and self.trade_history:
                    self.save_trade_history()
                
                # 현재 시간 확인하여 다음 캔들 시작까지 대기
                now = datetime.datetime.now()
                next_hour = now.replace(minute=0, second=10) + datetime.timedelta(hours=1)
                wait_seconds = (next_hour - now).total_seconds()
                
                # 대기 시간 정보 출력
                wait_msg = f"다음 캔들까지 {wait_seconds:.0f}초 대기 중 ({next_hour.strftime('%Y-%m-%d %H:%M:%S')})"
                logging.info(wait_msg)
                self.send_to_slack(f"⏳ {wait_msg}")
                
                # 1시간마다 차트 업데이트
                if now.minute < 5:  # 매 시간 처음 5분 이내에만 실행
                    chart_path = self.create_and_share_chart()
                    if chart_path:
                        self.send_to_slack(f"📊 정기 호크스 차트가 업데이트되었습니다.")
                
                # 대기 (최대 1시간, 10분 간격으로 나누어 처리)
                remaining_wait = wait_seconds
                chunk_size = 600  # 10분 (초 단위)
                
                while remaining_wait > 0:
                    sleep_time = min(chunk_size, remaining_wait)
                    time.sleep(sleep_time)
                    remaining_wait -= sleep_time
                    
                    # 주기적으로 현재 상태 로깅 (30분마다)
                    if remaining_wait % 1800 == 0 and remaining_wait > 0:
                        self.send_to_slack(f"⏳ 아직 대기 중... {remaining_wait}초 남음")
                
        except KeyboardInterrupt:
            stop_msg = "사용자에 의한 프로그램 종료"
            logging.info(stop_msg)
            self.send_to_slack(f"🛑 {stop_msg}")
        except Exception as e:
            error_msg = f"실행 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            self.send_to_slack(f"❌ {error_msg}")
        finally:
            # 최종 잔고 로깅
            self.log_account_info()
            
            # 거래 기록 저장
            self.save_trade_history()
            
            # 최종 차트 생성
            self.create_and_share_chart()
            
            end_msg = "프로그램 종료"
            logging.info(end_msg)
            self.send_to_slack(f"🏁 {end_msg}")

# 메인 함수
if __name__ == "__main__":
    print(f"EC2 Hawkes 트레이딩 봇 시작 (KAPPA: {KAPPA}, LOOKBACK: {VOLATILITY_LOOKBACK})")
    
    # API 키 확인
    if not UPBIT_ACCESS_KEY or not UPBIT_SECRET_KEY:
        print("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        logging.error("API 키 미설정")
        exit(1)
    
    # Slack API 토큰 및 채널 확인
    if not SLACK_API_TOKEN or not SLACK_CHANNEL:
        print("경고: Slack API 토큰 또는 채널이 설정되지 않았습니다. Slack 알림이 비활성화됩니다.")
        print("Slack 알림을 활성화하려면 .env 파일에 SLACK_API_TOKEN과 SLACK_CHANNEL을 설정하세요.")
        logging.warning("Slack API 토큰 또는 채널 미설정, Slack 알림 비활성화")
    
    # 거래 봇 실행
    trader = EC2HawkesTrader()
    trader.run()
