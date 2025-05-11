FROM python:3.9-slim

WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 복사
COPY . .

# pandas_ta 패치 적용 오류 방지를 위해 실행
RUN python fix_pandas_ta.py

# 필요한 디렉토리 생성
RUN mkdir -p /app/logs /app/charts

# 타임존 설정 (한국 시간)
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 실행 명령어
CMD ["python", "ec2_hawkes_live.py"]