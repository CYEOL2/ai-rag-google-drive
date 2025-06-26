# AI RAG Project

이 프로젝트는 Ollama, Qdrant, 그리고 RAG 애플리케이션을 Docker로 구성한 시스템입니다.

## 🚀 빠른 시작

### 전제 조건
1. **Google API 설정 필요** (Google Drive 연동용)
2. **Docker Desktop 설치 필요**

### 설정 단계
1. **Google API 인증 파일 준비** (아래 설정 가이드 참조)
2. **스크립트 실행**

**Windows:**
```cmd
scripts\quick-start.bat
```

**Linux/Mac:**
```bash
./scripts/quick-start.sh
```

> 💡 **이 하나의 명령어로 모든 서비스가 시작됩니다!**
> 
> - 이미 실행 중인 컨테이너는 그대로 유지
> - RAG 앱만 새로 빌드하고 시작
> - 첫 실행시 모델 다운로드 자동 진행

## ⚙️ Google Drive API 설정

### 1. Google Cloud Console 설정

1. **Google Cloud Console**에 접속: https://console.cloud.google.com/
2. **새 프로젝트 생성** 또는 기존 프로젝트 선택
3. **API 및 서비스 → 라이브러리**에서 다음 API 활성화:
   - Google Drive API
   - Google Docs API (선택사항)
4. **API 및 서비스 → 사용자 인증 정보**에서 **서비스 계정 생성**
5. **서비스 계정**에서 **키 생성** → **JSON 형식**으로 다운로드

### 2. 인증 파일 설정

다운로드한 JSON 파일을 **프로젝트 루트 경로**에 `service_account.json` 이름으로 저장:

```
AI-Project/
├── service_account.json  ← 여기에 저장!
├── app.py
├── docker-compose.yml
└── ...
```

> ⚠️ **보안 주의**: `service_account.json` 파일은 Git에 업로드되지 않도록 `.gitignore`에 포함되어 있습니다.

### 3. Google Drive 폴더 ID 설정

1. **Google Drive**에서 RAG 시스템이 사용할 폴더 생성
2. **폴더 URL**에서 **폴더 ID** 확인:
   ```
   https://drive.google.com/drive/folders/1j-IWG6JWndIa6yGSbChPH-k_jkoQ4OjZ
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          이 부분이 폴더 ID
   ```
3. **`docker-compose.full.yml`** 파일에서 폴더 ID 설정:
   ```yaml
   environment:
     - GOOGLE_DRIVE_FOLDER_ID=1j-IWG6JWndIa6yGSbChPH-k_jkoQ4OjZ  # 여기에 실제 폴더 ID 입력
   ```

### 4. 서비스 계정 권한 설정

1. **Google Drive 폴더**를 **서비스 계정 이메일**과 공유
2. **서비스 계정 이메일**은 `service_account.json` 파일의 `client_email` 필드에서 확인
3. **편집자** 권한으로 공유 (읽기/쓰기 필요)

## 📋 서비스 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Ollama | 11434 | AI 모델 서버 (llama3.2) |
| Qdrant | 6333, 6334 | 벡터 데이터베이스 |
| RAG App | 8000 | 웹 인터페이스 |

## 🤖 AI 호출 방법 비교

### 1. Ollama 직접 호출 vs RAG 시스템 호출

| 구분 | Ollama 직접 호출 | RAG 시스템 호출 |
|------|------------------|-----------------|
| **응답 방식** | 일반적인 AI 대화 | 문서 기반 정확한 답변 |
| **지식 범위** | 모델 학습 데이터만 | 업로드된 문서 + 모델 지식 |
| **최신성** | 학습 시점까지만 | 실시간 문서 업데이트 반영 |
| **정확성** | 일반적 수준 | 특정 도메인에서 높은 정확성 |
| **사용 목적** | 일반 대화, 창작 | 문서 검색, 정확한 정보 제공 |

### 2. Ollama 직접 호출 방법

#### 🔧 API 호출
```bash
# 기본 대화
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "prompt": "안녕하세요! AI에 대해 설명해주세요.",
    "stream": false
  }'

# 채팅 형식
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "user", "content": "파이썬으로 간단한 웹서버 만드는 방법을 알려주세요"}
    ],
    "stream": false
  }'
```

#### 🖥️ 명령줄에서 직접 사용
```bash
# Docker 컨테이너 내에서 실행
docker exec -it ollama ollama run llama3.2

# 대화 예시
>>> 안녕하세요!
>>> 파이썬 프로그래밍에 대해 알려주세요
>>> /bye  # 종료
```

#### 🌐 웹에서 접근
```
브라우저에서 접속: http://localhost:11434
```

### 3. RAG 시스템 호출 방법

#### 🌐 웹 인터페이스 (권장)
```
브라우저에서 접속: http://localhost:8000
```
- **특징**: 
  - 직관적인 채팅 인터페이스
  - 문서 업로드 및 동기화 기능
  - Google Drive 연동
  - 답변 출처 표시

#### 🔧 API 호출
```bash
# RAG 질문하기
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "우리 회사의 휴가 정책은 어떻게 되나요?"
  }'

# 문서 동기화
curl -X POST http://localhost:8000/sync

# 시스템 상태 확인
curl http://localhost:8000/health
```

#### 📊 Python에서 호출
```python
import requests

# RAG 시스템에 질문
response = requests.post('http://localhost:8000/ask', 
    json={'question': '프로젝트 일정은 언제까지인가요?'})
print(response.json())

# Ollama 직접 호출
response = requests.post('http://localhost:11434/api/generate',
    json={
        'model': 'llama3.2',
        'prompt': '일반적인 프로젝트 관리 방법론을 설명해주세요',
        'stream': False
    })
print(response.json())
```

### 4. 언제 어떤 방법을 사용할까?

#### 🎯 Ollama 직접 호출을 사용하는 경우:
- **일반적인 AI 대화**가 필요할 때
- **창작 활동** (시, 소설, 코드 생성 등)
- **일반 상식** 질문
- **브레인스토밍**이나 아이디어 생성
- **프로그래밍 도움** (일반적인 코딩 질문)

```bash
# 예시: 창작 활동
docker exec -it ollama ollama run llama3.2
>>> "미래 도시에 대한 SF 소설을 써주세요"
```

#### 📚 RAG 시스템을 사용하는 경우:
- **특정 문서**에서 정보를 찾을 때
- **회사 내부 정책**이나 규정 질문
- **기술 문서** 검색
- **정확한 사실** 확인이 필요할 때
- **최신 업데이트된 정보**가 필요할 때

```
웹 브라우저: http://localhost:8000
질문: "우리 회사 보안 정책에서 비밀번호 규칙은 무엇인가요?"
→ 업로드된 보안 정책 문서에서 정확한 답변 제공
```

## 🔧 주요 특징

### 간편한 실행
- **원클릭 시작**: 하나의 스크립트로 모든 서비스 시작
- **지능적 관리**: 실행 중인 서비스는 유지, 필요한 것만 시작
- **자동 빌드**: RAG 앱 이미지 자동 빌드 및 업데이트

### 안정적 운영
- **분리된 실행**: 모델 다운로드 중 RAG 앱 오류 방지
- **헬스체크**: 서비스 준비 상태 자동 확인
- **PowerShell 호환**: Windows 환경에서 안정적 동작

## 📝 사용법

### 일반적인 사용 (권장)
```cmd
# Windows
scripts\quick-start.bat

# Linux/Mac
./scripts/quick-start.sh
```

### 전체 재시작이 필요한 경우
```cmd
# 모든 서비스 중지 후 재시작
docker-compose -f docker-compose.full.yml down
scripts\quick-start.bat
```

### 완전 초기화 (모델 재다운로드)
```cmd
# 모든 데이터 삭제 후 재시작
docker-compose -f docker-compose.full.yml down -v
scripts\quick-start.bat
```

### 서비스 중지
```cmd
# 모든 서비스 중지
docker-compose -f docker-compose.full.yml down

# RAG 앱만 중지
docker stop rag-app
```

## 📁 프로젝트 구조

```
AI-Project/
├── scripts/
│   ├── quick-start.bat           # Windows용 통합 실행 스크립트
│   └── quick-start.sh            # Linux/Mac용 통합 실행 스크립트
├── docker-compose.yml            # 기본 인프라 (Ollama + Qdrant)
├── docker-compose.full.yml       # 전체 서비스 (실제 사용)
├── Dockerfile                    # RAG 앱 Docker 이미지
├── app.py                        # RAG 애플리케이션
├── requirements.txt              # Python 의존성
├── service_account.json          # Google Drive API 인증 (필수!)
├── .gitignore                    # Git 무시 파일
├── ollama/
│   ├── Dockerfile               # Ollama 커스텀 이미지
│   └── entrypoint.sh            # 모델 자동 다운로드 스크립트
└── documents/                   # 문서 저장 폴더
```

## 🛠️ 문제 해결

### 설정 관련 문제

#### 1. service_account.json 파일 없음
```
ERROR: service_account.json file not found.
```
**해결방법:**
- Google Cloud Console에서 서비스 계정 키를 JSON 형식으로 다운로드
- 파일명을 `service_account.json`으로 변경
- 프로젝트 루트 경로에 저장

#### 2. Google Drive 폴더 접근 권한 오류
```
ERROR: Permission denied to access Google Drive folder
```
**해결방법:**
- `service_account.json`의 `client_email`을 Google Drive 폴더에 공유
- 편집자 권한으로 설정
- `docker-compose.full.yml`의 `GOOGLE_DRIVE_FOLDER_ID` 값 확인

#### 3. 폴더 ID 오류
```
ERROR: Invalid folder ID
```
**해결방법:**
- Google Drive 폴더 URL에서 정확한 폴더 ID 복사
- `docker-compose.full.yml`의 `GOOGLE_DRIVE_FOLDER_ID` 값 업데이트

### 일반적인 문제들

#### 1. 첫 실행시 시간이 오래 걸림
- llama3.2 모델 다운로드로 5-10분 소요 (정상)
- 인터넷 연결 상태 확인
- 충분한 디스크 공간 확보 (최소 4GB)

#### 2. RAG 앱이 시작되지 않음
```cmd
# Qdrant 상태 확인
docker logs qdrant

# Ollama 상태 확인  
docker logs ollama

# 서비스 재시작
docker-compose -f docker-compose.full.yml restart
```

#### 3. 포트 충돌
```cmd
# 포트 사용 현황 확인
netstat -an | findstr "8000\|11434\|6333"

# 다른 애플리케이션 종료 후 재시작
scripts\quick-start.bat
```

### 로그 확인 방법
```cmd
# 실시간 로그
docker logs -f rag-app
docker logs -f ollama
docker logs -f qdrant

# 전체 서비스 로그
docker-compose -f docker-compose.full.yml logs -f

# 최근 로그만
docker logs --tail 50 rag-app
```

## 🔄 개발 워크플로우

### 코드 수정 후 재시작
```cmd
# RAG 앱만 재시작 (가장 빠름)
scripts\quick-start.bat
```

### Docker 이미지 강제 재빌드
```cmd
# 캐시 없이 새로 빌드
docker-compose -f docker-compose.full.yml build --no-cache rag-app
scripts\quick-start.bat
```

## 🌐 접속 정보

- **RAG 웹 인터페이스**: http://localhost:8000
- **Ollama API**: http://localhost:11434  
- **Qdrant 대시보드**: http://localhost:6333/dashboard

## 📊 시스템 요구사항

- **Docker Desktop**: 최신 버전
- **메모리**: 최소 8GB RAM 권장
- **디스크**: 최소 10GB 여유 공간
- **네트워크**: 모델 다운로드를 위한 안정적인 인터넷 연결
- **Google Cloud 계정**: Google Drive API 사용을 위한 서비스 계정

## 🔐 환경 변수

RAG 앱에서 사용하는 주요 환경 변수:

```bash
OLLAMA_BASE_URL=http://ollama:11434
QDRANT_HOST=qdrant
QDRANT_PORT=6333
GOOGLE_DRIVE_FOLDER_ID=1j-IWG6JWndIa6yGSbChPH-k_jkoQ4OjZ  # 실제 폴더 ID로 변경
SYNC_INTERVAL_MINUTES=30
```

> ⚠️ **중요**: `GOOGLE_DRIVE_FOLDER_ID`는 반드시 실제 Google Drive 폴더 ID로 변경해야 합니다.

## 💡 사용 팁

1. **일상적 사용**: `scripts\quick-start.bat` 하나로 충분
2. **첫 실행**: 모델 다운로드 시간 고려하여 여유있게 대기
3. **Google Drive 동기화**: 웹 인터페이스에서 동기화 버튼으로 최신 문서 반영
4. **문제 발생시**: 전체 재시작 → `docker-compose -f docker-compose.full.yml down && scripts\quick-start.bat`
5. **로그 확인**: 스크립트 실행 후 실시간 로그 보기 옵션 활용

## 🎯 요약

**설정 준비:**
1. `service_account.json` 파일을 루트에 저장
2. `docker-compose.full.yml`에서 `GOOGLE_DRIVE_FOLDER_ID` 설정
3. Google Drive 폴더를 서비스 계정과 공유

**모든 것을 한 번에 시작:**
```cmd
scripts\quick-start.bat
```

**AI 호출 방법:**
- **일반 대화**: http://localhost:11434 (Ollama 직접)
- **문서 기반 질답**: http://localhost:8000 (RAG 시스템)

**서비스 중지:**
```cmd
docker-compose -f docker-compose.full.yml down
```

이제 Google Drive 연동이 포함된 완전한 RAG 시스템을 하나의 명령어로 실행할 수 있습니다! 🚀
