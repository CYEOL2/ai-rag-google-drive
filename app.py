# app.py (Final Corrected Version)

import uvicorn
import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import uuid
from pydantic import BaseModel, Field

# LangChain 및 관련 라이브러리
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings

# 대체 import 시도
try:
    from langchain_qdrant import Qdrant
except ImportError:
    from langchain_community.vectorstores import Qdrant
    
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

# Google Drive API 관련
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# 기타
import io
import pandas as pd
from qdrant_client import QdrantClient, models
from apscheduler.schedulers.background import BackgroundScheduler

# --- 1. 로깅 및 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 환경 변수 및 주요 설정값 ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama3.2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
COLLECTION_NAME = "gdrive_rag_collection"
SERVICE_ACCOUNT_FILE = "/app/service_account.json"
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
SYNC_INTERVAL_MINUTES = int(os.getenv("SYNC_INTERVAL_MINUTES", "60"))
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# RAG 처리 관련 설정 - Excel에 최적화
CHUNK_SIZE = 2000  # 업무공유 문서 특성상 더 큰 청크
CHUNK_OVERLAP = 300  # 더 많은 오버랩으로 컨텍스트 보존
EMBEDDING_BATCH_SIZE = 128
CHUNK_VERSION = f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_excel_optimized"

# --- 3. 전역 변수 ---
vectorstore = None
qa_chain = None
embeddings = None
scheduler = BackgroundScheduler(timezone="Asia/Seoul")
drive_service = None

# --- 4. Google Drive 연동 ---
def get_drive_service():
    global drive_service
    if drive_service: return drive_service
    if not Path(SERVICE_ACCOUNT_FILE).exists():
        logger.error(f"서비스 계정 키 파일({SERVICE_ACCOUNT_FILE})을 찾을 수 없습니다.")
        return None
    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        drive_service = build('drive', 'v3', credentials=credentials)
        logger.info("Google Drive API 서비스 인증 완료")
        return drive_service
    except Exception as e:
        logger.error(f"Google Drive API 서비스 인증 실패: {e}")
        return None

def _decode_bytes_to_text(content_bytes, file_id):
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'iso-8859-1', 'latin1']
    for encoding in encodings:
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    logger.warning(f"파일 {file_id} 디코딩 실패. 일부 문자 대체됨.")
    return content_bytes.decode('utf-8', errors='replace')

def download_file_content(service, file_id, file_name, mime_type):
    request = None
    if mime_type == 'application/vnd.google-apps.document':
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
    elif mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        request = service.files().get_media(fileId=file_id)
    elif mime_type.startswith('text/'):
        request = service.files().get_media(fileId=file_id)
    else:
        logger.warning(f"지원하지 않는 MIME 타입({mime_type})으로 파일 '{file_name}'을(를) 건너뜁니다.")
        return None

    try:
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done: _, done = downloader.next_chunk()
        file_io.seek(0)

        # 엑셀 파일인 경우, 업무공유 문서에 특화된 파싱
        if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
            try:
                xls = pd.ExcelFile(file_io, engine='openpyxl')
                parts = []
                logger.info(f"엑셀 파일 '{file_name}' 처리 시작. 시트 목록: {xls.sheet_names}")
                
                for sheet in xls.sheet_names:
                    # 모든 셀을 문자열로 읽고 병합된 셀 정보 유지
                    df = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str, keep_default_na=False)
                    
                    if df.empty: 
                        continue
                    
                    # 시트 헤더
                    header = f"\n{'='*60}\n"
                    header += f"📋 엑셀 시트: {sheet}\n"
                    header += f"{'='*60}\n\n"
                    
                    # 셀별 상세 정보 추출 (행렬 구조 보존)
                    content_sections = []
                    
                    for row_idx, row in df.iterrows():
                        row_content = []
                        for col_idx, cell_value in enumerate(row):
                            if pd.notna(cell_value) and str(cell_value).strip():
                                clean_val = str(cell_value).strip()
                                if clean_val and clean_val.lower() not in ['nan', 'none', 'unnamed']:
                                    row_content.append(f"[{row_idx+1},{col_idx+1}] {clean_val}")
                        
                        if row_content:
                            content_sections.append("\n".join(row_content))
                    
                    # 프로젝트별 상세 정보 추출
                    all_text = " ".join([str(cell) for row in df.values for cell in row if pd.notna(cell)])
                    
                    # 주요 정보 패턴 검색 (범용적)
                    project_info = "\n🔍 주요 정보:\n"
                    
                    # 프로젝트/사업 관련 패턴 검색
                    import re
                    project_patterns = re.findall(r'[가-힣\w\s]*(?:프로젝트|사업|시스템|개발|고도화)[가-힣\w\s]*', all_text, re.IGNORECASE)
                    if project_patterns:
                        unique_projects = list(set([p.strip() for p in project_patterns if len(p.strip()) > 2]))
                        project_info += "\n📌 프로젝트/사업:\n"
                        for pattern in unique_projects[:10]:
                            project_info += f"  - {pattern}\n"
                    
                    # 인력/담당자 관련 패턴 검색
                    manpower_patterns = re.findall(r'(?:PL|투입|인력|담당)[^\n]*(?:명|개월|년)[^\n]*', all_text, re.IGNORECASE)
                    if manpower_patterns:
                        project_info += "\n👥 인력 정보:\n"
                        for pattern in manpower_patterns:
                            project_info += f"  - {pattern.strip()}\n"
                    
                    # 담당자명 패턴 검색
                    name_patterns = re.findall(r'[가-힣]{2,4}\s*(?:부장|팀장|연구원|개발자|대리|과장|차장)?', all_text)
                    if name_patterns:
                        unique_names = list(set([name.strip() for name in name_patterns if len(name.strip()) >= 2]))
                        if unique_names:
                            project_info += f"\n👤 관련 인명: {', '.join(unique_names[:15])}\n"
                    
                    # 일정 정보 검색
                    schedule_patterns = re.findall(r'\d{4}[.-]\d{1,2}[.-]\d{1,2}|\d{1,2}월\s*착수|\d{1,2}개월|\d{4}년\s*\d{1,2}월', all_text)
                    if schedule_patterns:
                        project_info += f"\n📅 일정 정보: {', '.join(set(schedule_patterns[:10]))}\n"
                    
                    # 최종 컨텐츠 조합
                    sheet_content = header + "\n".join(content_sections[:50]) + project_info  # 최대 50개 섹션
                    parts.append(sheet_content)
                    logger.info(f"시트 '{sheet}' 처리 완료: {len(content_sections)}개 섹션")
                
                result = "\n\n".join(parts) if parts else None
                if result:
                    logger.info(f"엑셀 파일 '{file_name}' 처리 완료. 총 {len(parts)}개 시트, 길이: {len(result)}자")
                return result
            except Exception as excel_error:
                logger.error(f"엑셀 파일 처리 중 오류 ({file_name}): {excel_error}")
                return None

        # 그 외 텍스트 기반 파일 처리
        return _decode_bytes_to_text(file_io.getvalue(), file_id)
    except Exception as e:
        logger.error(f"파일 다운로드 또는 파싱 실패 ({file_name}): {e}")
        return None

def get_files_from_drive(folder_id):
    service = get_drive_service()
    if not service: return []
    try:
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name, modifiedTime, mimeType)", pageSize=1000).execute()
        docs = []
        for file in results.get('files', []):
            logger.info(f"처리 대상 파일 발견: {file['name']} (ID: {file['id']})")
            if content := download_file_content(service, file['id'], file['name'], file['mimeType']):
                docs.append({'id': file['id'], 'name': file['name'], 'content': content, 'modified_time': file['modifiedTime']})
                logger.info(f"파일 콘텐츠 로드 완료: {file['name']}")
        return docs
    except HttpError as e:
        logger.error(f"Google Drive API 오류: {e}")
        return []

# --- 5. 동기화 로직 (파일 수정일시 기준 중복 방지) ---
def perform_sync():
    global embeddings, vectorstore
    if not all([embeddings, vectorstore]) or not GOOGLE_DRIVE_FOLDER_ID:
        logger.warning("RAG 컴포넌트 미초기화 또는 폴더 ID 누락으로 동기화를 건너뜁니다.")
        return

    logger.info(f"--- 동기화 시작 (폴더 ID: {GOOGLE_DRIVE_FOLDER_ID}) ---")
    documents = get_files_from_drive(GOOGLE_DRIVE_FOLDER_ID)
    if not documents:
        logger.info("동기화할 문서가 없습니다.")

    qdrant_client = vectorstore.client
    processed_files, skipped_files = 0, 0
    drive_file_ids = {doc['id'] for doc in documents}

    for doc in documents:
        file_id, file_name, modified_time, content = doc['id'], doc['name'], doc['modified_time'], doc['content']
        
        # Vector DB에서 해당 파일의 최신 메타데이터 확인
        existing, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME, limit=1, with_vectors=False,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]),
            with_payload=["modified_time", "chunk_version"]
        )
        stored_payload = existing[0].payload if existing else None
        
        # 조건 확인: 파일 수정일시와 청킹 버전이 모두 동일한 경우 건너뛰기
        if (stored_payload and
            stored_payload.get("modified_time") == modified_time and
            stored_payload.get("chunk_version") == CHUNK_VERSION):
            skipped_files += 1
            continue

        status = "New"
        if stored_payload:
            if stored_payload.get("chunk_version") != CHUNK_VERSION:
                status = "Update (Chunk Strategy Changed)"
            else:
                status = "Update (File Modified)"
        
        logger.info(f"  - [{status}] 파일 처리 시작: {file_name}")
        
        # 파일 내용 분할 (Chunking)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(content)
        metadata = {"source": file_name, "file_id": file_id, "modified_time": modified_time, "chunk_version": CHUNK_VERSION}
        docs_to_embed = [Document(page_content=t, metadata=metadata) for t in chunks]
        
        # 임베딩 및 Vector DB 저장을 위한 데이터 생성
        points, batch_failed = [], False
        for i in range(0, len(docs_to_embed), EMBEDDING_BATCH_SIZE):
            batch = docs_to_embed[i:i + EMBEDDING_BATCH_SIZE]
            try:
                vectors = embeddings.embed_documents([d.page_content for d in batch])
                for k, doc_chunk in enumerate(batch):
                    pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_id}-{i+k}"))
                    payload = {**doc_chunk.metadata, 'page_content': doc_chunk.page_content}
                    points.append(models.PointStruct(id=pid, vector=vectors[k], payload=payload))
            except Exception as e:
                logger.error(f"임베딩 배치 처리 실패 ({file_name}, 시작 인덱스: {i}): {e}")
                batch_failed = True; break
        if batch_failed: continue
        
        # 업데이트인 경우, 기존 데이터 먼저 삭제
        if status.startswith("Update"):
            logger.debug(f"기존 데이터 삭제 중: {file_name}")
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]))
            )

        if points:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        processed_files += 1

    # Google Drive에서 삭제된 파일들을 Vector DB에서 제거
    logger.info("삭제된 파일 확인 및 정리 시작...")
    qdrant_file_ids = set()
    next_offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME, limit=256, offset=next_offset,
            with_payload=['file_id'], with_vectors=False
        )
        for point in points: qdrant_file_ids.add(point.payload['file_id'])
        if not next_offset: break
    
    ids_to_delete = qdrant_file_ids - drive_file_ids
    if ids_to_delete:
        logger.info(f"총 {len(ids_to_delete)}개의 삭제된 파일을 Qdrant에서 제거합니다.")
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="file_id", match=models.MatchAny(any=list(ids_to_delete)))]
            ))
        )
    
    logger.info(f"--- 동기화 완료: 처리 {processed_files}개, 건너뜀 {skipped_files}개, 삭제 {len(ids_to_delete)}개 (Chunk Version: {CHUNK_VERSION}) ---")

# --- 6. FastAPI 수명주기 (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, qa_chain, embeddings, scheduler
    logger.info("애플리케이션 시작...")

    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vector_size = len(embeddings.embed_query("test"))
        logger.info(f"Ollama 임베딩 모델({EMBEDDING_MODEL}) 로드 완료. 벡터 차원: {vector_size}")
    except Exception as e:
        logger.critical(f"임베딩 모델 초기화 실패! 앱을 시작할 수 없습니다. 오류: {e}")
        raise RuntimeError("Embedding model initialization failed") from e

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        current_vector_size = info.vectors_config.params.size
        if current_vector_size != vector_size:
            logger.warning(f"컬렉션의 벡터 차원({current_vector_size})이 모델({vector_size})과 달라 재생성합니다.")
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
    except Exception:
        logger.info(f"컬렉션 '{COLLECTION_NAME}'을(를) 새로 생성합니다.")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )

    vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Excel 데이터에 최적화된 retriever 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={'k': 12}  # 더 많은 컨텍스트 수집
    )
    
    # Excel 문서 해석에 특화된 프롬프트
    custom_prompt = PromptTemplate(
        template="""당신은 엑셀 문서를 분석하는 전문가입니다.

[중요 지침]
1. 제공된 문서 내용만을 정확히 기반으로 답변하세요
2. 셀 위치 정보 [행,열]을 참고하여 관련 데이터를 종합 분석하세요
3. 프로젝트명, 담당자, 인력, 일정 등 구체적 정보를 정확히 인용하세요
4. 여러 시트에 걸친 관련 정보가 있다면 모두 종합하여 완전한 답변을 제공하세요
5. 베트남어나 기타 언어를 절대 사용하지 말고 오직 한국어로만 답변하세요
6. 추측하지 말고 문서에 명확히 기록된 정보만 제공하세요

[분석할 문서 내용]
{context}

[질문]
{question}

[정확한 한국어 답변]""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    logger.info("RAG 컴포넌트 초기화 완료.")

    # 앱 시작 시 즉시 1회 동기화, 이후 주기적으로 동기화
    if not scheduler.running:
        scheduler.add_job(perform_sync, 'date', id="gdrive_initial_sync")
        scheduler.add_job(perform_sync, 'interval', minutes=SYNC_INTERVAL_MINUTES, id="gdrive_sync_job")
        scheduler.start()
        logger.info(f"자동 동기화 스케줄러 시작. 최초 실행 후 {SYNC_INTERVAL_MINUTES}분마다 실행됩니다.")

    yield

    logger.info("애플리케이션 종료...")
    if scheduler.running: scheduler.shutdown()

# --- 7. FastAPI 앱 및 API ---
app = FastAPI(title="Google Drive RAG Application", version="1.5.0-fixed", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, title="질문", description="RAG 모델에게 할 질문")

@app.get("/")
async def root():
    return {"message": "Google Drive RAG API가 실행 중입니다. /docs 에서 API 문서를 확인하세요."}

@app.post("/query")
async def query_documents(req: QueryRequest):
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA 체인이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.")
    try:
        question = req.question.strip()
        result = qa_chain.invoke({"query": question})
        
        # 깔끔한 응답 (source_documents 배열 제거)
        return {
            "question": question, 
            "answer": result['result'].strip()
        }
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="질문 처리 중 서버 오류가 발생했습니다.")

@app.post("/manual-sync", status_code=202)
async def manual_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(perform_sync)
    return {"message": "수동 동기화 작업이 백그라운드에서 시작되었습니다. 서버 로그를 확인하세요."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "drive_service": "connected" if drive_service else "not_connected",
        "rag_components_initialized": qa_chain is not None,
        "scheduler_running": scheduler.running if scheduler else False
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)