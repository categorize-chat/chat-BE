FROM python:3.9-slim

WORKDIR /app

# 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 필요한 파일 복사
COPY model.py .
COPY .env .

# 포트 노출
EXPOSE 5000

# 컨테이너 시작 시 실행할 명령
CMD ["python", "model.py"] 