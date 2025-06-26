# Dockerfile (Corrected)

FROM python:3.11-slim

WORKDIR /app

# requirements.txt를 먼저 복사하여 빌드 캐시 활용
COPY ./requirements.txt /app/requirements.txt

# pip 업그레이드 및 의존성 설치
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# 나머지 코드를 복사 (이제 볼륨 마운트가 제한적이므로 이 단계가 의미 있음)
COPY . /app/

EXPOSE 8000

# [핵심 수정] CMD에서 파일 경로를 명시적으로 app.py로 수정
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
