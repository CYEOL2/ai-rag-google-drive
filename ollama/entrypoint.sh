#!/bin/sh

# 스크립트 내에서 오류 발생 시 즉시 중단
set -e

# 종료 신호(SIGTERM, SIGINT)를 받았을 때 실행할 함수 정의
cleanup() {
  echo "종료 신호를 수신했습니다. Ollama 서버를 종료합니다..."
  if [ -n "$pid" ]; then
    kill -TERM "$pid"
    wait "$pid"
  fi
  echo "Ollama 서버가 종료되었습니다."
}

# SIGTERM과 SIGINT 신호에 대해 cleanup 함수를 실행하도록 설정
trap 'cleanup' TERM INT

# Ollama 서버를 백그라운드에서 실행
echo "Ollama 서버를 백그라운드에서 시작합니다..."
/bin/ollama serve &
pid=$!

# [중요] 서버가 실제로 준비될 때까지 기다리는 로직이 필요하지만,
# curl이 없으므로 단순한 sleep으로 대체하거나, healthcheck의 start_period에 의존합니다.
# entrypoint.sh의 복잡성을 줄이기 위해, 여기서는 단순 대기 후 pull을 시도합니다.
echo "Ollama 서버 초기화 대기 중..."
sleep 15 # 서버가 시작될 시간을 충분히 줍니다.

# 필요한 모델이 이미 있는지 확인하고, 없을 때만 다운로드
MODEL_NAME="llama3.2"
if ollama list | grep -q "$MODEL_NAME"; then
  echo "모델 '$MODEL_NAME'이(가) 이미 존재합니다. 다운로드를 건너뜁니다."
else
  echo "모델 '$MODEL_NAME' 다운로드를 시작합니다..."
  ollama pull "$MODEL_NAME"
  echo "모델 다운로드가 완료되었습니다."
fi

# Healthcheck를 위한 신호 파일 생성
touch /tmp/models_ready
echo "Healthcheck 신호 파일(/tmp/models_ready) 생성 완료."

echo "Ollama가 모든 설정을 마쳤습니다. 이제 요청을 받을 수 있습니다."

# 백그라운드 서버 프로세스가 종료될 때까지 대기
wait "$pid"