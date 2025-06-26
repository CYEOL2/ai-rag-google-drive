# AI RAG Project

**llama3.2 모델**과 **Google Drive 연동**을 지원하는 RAG (Retrieval-Augmented Generation) 시스템입니다.

## 🤖 AI 모델 정보

- **기본 모델**: llama3.2 (Meta의 최신 언어 모델)
- **자동 설치**: 첫 실행시 자동 다운로드 (약 5-10분 소요)
- **용도**: 문서 기반 질의응답 및 일반 AI 대화

## 🚀 빠른 시작

### 전제 조건
1. **Docker Desktop 설치**
2. **Google API 설정** (아래 가이드 참조)
3. **디스크 공간 10GB 이상**

### 실행 순서

#### 1단계: 기본 인프라 시작
```bash
docker-compose up -d
```
> ⏳ **첫 실행시**: llama3.2 모델 다운로드로 5-10분 소요

#### 2단계: 모델 다운로드 완료 확인
```bash
# 모델 준비 상태 확인
docker logs ollama
# "Ollama가 모든 설정을 마쳤습니다" 메시지 확인
```

#### 3단계: RAG 앱 시작
**Windows:**
```cmd
scripts\quick-start.bat
```

**Linux/Mac:**
```bash
./scripts/quick-start.sh
```

#### 4단계: 접속
- **문서 기반 질답**: http://localhost:8000
- **일반 AI 대화**: http://localhost:11434

## ⚙️ Google Drive API 설정

### 1. Google Cloud Console 설정
1. https://console.cloud.google.com/ 접속
2. **새 프로젝트 생성**
3. **Google Drive API 활성화**
4. **서비스 계정 생성** 및 **JSON 키 다운로드**

### 2. 파일 배치
```
AI-Project/
├── service_account.json  ← 다운로드한 JSON 파일을 이 이름으로 저장
└── docker-compose.full.yml
```

### 3. 폴더 ID 설정
1. **Google Drive에서 폴더 생성**
2. **폴더 URL에서 ID 복사**:
   ```
   https://drive.google.com/drive/folders/1j-IWG6JWndIa6yGSbChPH-k_jkoQ4OjZ
                                          ↑ 이 부분이 폴더 ID
   ```
3. **`docker-compose.full.yml` 수정**:
   ```yaml
   environment:
     - GOOGLE_DRIVE_FOLDER_ID=실제폴더ID입력
   ```

### 4. 권한 설정
- **Google Drive 폴더**를 **서비스 계정 이메일**과 공유 (편집자 권한)
- 서비스 계정 이메일은 `service_account.json`의 `client_email` 확인

## 🤖 AI 사용법

### 문서 기반 질답 (RAG)
```bash
# 웹 인터페이스 (권장)
http://localhost:8000

# API 호출
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "회사 정책에 대해 알려주세요"}'
```

### 일반 AI 대화 (llama3.2 직접)
```bash
# 웹 접속
http://localhost:11434

# 명령줄
docker exec -it ollama ollama run llama3.2

# API 호출
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "prompt": "안녕하세요", "stream": false}'
```

## 📋 서비스 포트

| 서비스 | 포트 | 용도 |
|--------|------|------|
| RAG App | 8000 | 문서 기반 질답 |
| Ollama | 11434 | 직접 AI 대화 |
| Qdrant | 6333 | 벡터 DB 대시보드 |

## 🔄 일반적인 사용법

### 매일 사용
```cmd
# 모든 서비스 한번에 시작
scripts\quick-start.bat
```

### 전체 재시작
```cmd
# 서비스 중지
docker-compose -f docker-compose.full.yml down

# 다시 시작
docker-compose up -d
# 모델 준비 대기 후...
scripts\quick-start.bat
```

### 완전 초기화 (모델 재다운로드)
```cmd
docker-compose -f docker-compose.full.yml down -v
docker-compose up -d
```

## 🛠️ 문제 해결

### Google Drive 오류
```bash
# 권한 확인
docker logs rag-app | grep "Google Drive"

# 폴더 ID 확인
docker exec rag-app env | grep GOOGLE_DRIVE_FOLDER_ID
```

### 모델 다운로드 문제
```bash
# 진행 상황 확인
docker logs -f ollama

# 수동 다운로드
docker exec -it ollama ollama pull llama3.2
```

### 포트 충돌
```bash
# 사용 중인 포트 확인
netstat -an | findstr "8000\|11434\|6333"
```

## 📁 주요 파일

```
AI-Project/
├── scripts/quick-start.bat       # Windows 실행 스크립트
├── scripts/quick-start.sh        # Linux/Mac 실행 스크립트  
├── docker-compose.yml            # 기본 인프라
├── docker-compose.full.yml       # 전체 서비스
├── service_account.json          # Google API 키 (필수)
├── app.py                        # RAG 애플리케이션
└── ollama/entrypoint.sh          # llama3.2 자동 설치
```

## 💡 핵심 팁

1. **첫 실행**: 인터넷 연결 좋은 환경에서 모델 다운로드
2. **일상 사용**: `scripts\quick-start.bat` 하나면 충분
3. **문서 업데이트**: Google Drive 폴더에 파일 추가 후 웹에서 동기화
4. **문제 발생**: 전체 재시작이 가장 확실한 해결책

## 🎯 요약

```bash
# 1. 설정
service_account.json 준비 + 폴더 ID 설정

# 2. 실행  
docker-compose up -d          # 기본 인프라 + 모델 다운로드
scripts\quick-start.bat       # RAG 앱 시작

# 3. 사용
http://localhost:8000         # 문서 기반 질답
http://localhost:11434        # 일반 AI 대화

# 4. 종료
docker-compose -f docker-compose.full.yml down
```

🚀 **Google Drive의 문서를 AI가 학습하여 정확한 답변을 제공하는 RAG 시스템입니다!**
