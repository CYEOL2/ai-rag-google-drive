# ollama/Dockerfile (Final Simple Version)

FROM ollama/ollama

# entrypoint.sh 스크립트를 컨테이너 내부로 복사 및 권한 설정
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]