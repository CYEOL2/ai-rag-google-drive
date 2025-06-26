# app.py (Final Version - No changes needed with the correct environment)

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
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

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

# RAG 처리 관련 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_BATCH_SIZE = 128

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
    for encoding in ('utf-8', 'cp949'):
        try: return content_bytes.decode(encoding)
        except UnicodeDecodeError: continue
    logger.error(f"파일 인코딩을 해석할 수 없습니다: {file_id}")
    return None

def download_file_content(service, file_id, file_name, mime_type):
    request = None
    if mime_type == 'application/vnd.google-apps.document':
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
    elif mime_type == 'application/vnd.google-apps.spreadsheet':
        request = service.files().export_media(fileId=file_id, mimeType='text/csv')
    elif mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        request = service.files().get_media(fileId=file_id)
    elif mime_type.startswith('text/'):
        request = service.files().get_media(fileId=file_id)
    else:
        return None

    try:
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done: _, done = downloader.next_chunk()
        file_io.seek(0)

        if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
            xls = pd.ExcelFile(file_io, engine='openpyxl')
            parts = []
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, dtype=str).dropna(how='all').dropna(axis=1, how='all')
                if df.empty: continue
                header = f"--- 시트: {sheet} ---\n"
                rows = [", ".join(f"{col}: {val}" for col, val in row.dropna().items()) for _, row in df.iterrows()]
                if content := "\n".join(filter(None, rows)):
                    parts.append(header + content)
            return "\n\n".join(parts) if parts else None

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
            if content := download_file_content(service, file['id'], file['name'], file['mimeType']):
                docs.append({'id': file['id'], 'name': file['name'], 'content': content, 'modified_time': file['modifiedTime']})
                logger.info(f"파일 로드 완료: {file['name']}")
        return docs
    except HttpError as e:
        logger.error(f"Google Drive API 오류: {e}")
        return []

# --- 5. 동기화 로직 (파일 삭제 처리 기능 포함) ---
def perform_sync():
    global embeddings, vectorstore
    if not all([embeddings, vectorstore]) or not GOOGLE_DRIVE_FOLDER_ID:
        logger.warning("RAG 컴포넌트 미초기화 또는 폴더 ID 누락으로 동기화를 건너뜁니다.")
        return

    logger.info(f"--- 동기화 시작 (폴더 ID: {GOOGLE_DRIVE_FOLDER_ID}) ---")
    documents = get_files_from_drive(GOOGLE_DRIVE_FOLDER_ID)

    qdrant_client = vectorstore.client
    processed_files, skipped_files = 0, 0
    drive_file_ids = {doc['id'] for doc in documents}

    for doc in documents:
        file_id, file_name, modified_time, content = doc['id'], doc['name'], doc['modified_time'], doc['content']
        existing, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME, limit=1, with_vectors=False,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]),
            with_payload=["modified_time"]
        )
        stored_payload = existing[0].payload if existing else None
        if stored_payload and stored_payload.get("modified_time") == modified_time:
            skipped_files += 1
            continue
        status = "Update" if stored_payload else "New"
        logger.info(f"  - [{status}] 파일 처리 시작: {file_name}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(content)
        metadata = {"source": file_name, "file_id": file_id, "modified_time": modified_time}
        docs_to_embed = [Document(page_content=t, metadata=metadata) for t in chunks]
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
                logger.error(f"    -> 임베딩 배치 처리 실패 ({file_name}, 시작 인덱스: {i}): {e}")
                batch_failed = True; break
        if batch_failed: continue
        if status == "Update":
            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))])))
        if points:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        processed_files += 1

    logger.info("--- 삭제된 파일 확인 및 정리 시작 ---")
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
    
    logger.info(f"--- 동기화 완료: 처리 {processed_files}개, 건너뜀 {skipped_files}개, 삭제 {len(ids_to_delete)}개 ---")

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
        # 컬렉션 정보를 가져오는 것을 시도합니다.
        info = client.get_collection(collection_name=COLLECTION_NAME)
        
        # [수정 1] 구버전/신버전 라이브러리 모두 호환되도록 벡터 크기 확인
        # hasattr를 사용하여 안전하게 속성 존재 여부를 확인합니다.
        current_vector_size = -1
        if hasattr(info, 'vectors_config'): # 신버전 경로
            current_vector_size = info.vectors_config.params.size
        elif hasattr(info, 'config'): # 구버전 경로
            current_vector_size = info.config.params.vectors.size

        if current_vector_size != vector_size:
            logger.warning("컬렉션의 벡터 차원이 모델과 달라 재생성합니다.")
            # [수정 2] 버전 호환성을 위해 wait 인자 없이 호출
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )

    except Exception as e:
        # get_collection이 404 에러 등을 포함한 예외를 발생시키면, 컬렉션이 없는 것으로 간주합니다.
        if "404" in str(e) or "Not found" in str(e) or "doesn't exist" in str(e):
             logger.info(f"컬렉션 '{COLLECTION_NAME}'을(를) 찾을 수 없어 새로 생성합니다.")
        else:
             logger.warning(f"컬렉션 확인 중 예외 발생({e}). 새로 생성합니다.")
        
        # [수정 2] 버전 호환성을 위해 wait 인자 없이 호출
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )

    # (이하 코드는 동일합니다)
    vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    logger.info("RAG 컴포넌트 초기화 완료.")

    if not scheduler.running:
        scheduler.add_job(perform_sync, 'interval', minutes=SYNC_INTERVAL_MINUTES, id="gdrive_sync_job", replace_existing=True)
        scheduler.add_job(perform_sync, 'date', id="gdrive_initial_sync", replace_existing=True)
        scheduler.start()
        logger.info(f"자동 동기화 스케줄러 시작. {SYNC_INTERVAL_MINUTES}분마다 실행됩니다.")

    yield

    logger.info("애플리케이션 종료...")
    if scheduler.running: scheduler.shutdown()


# --- 7. FastAPI 앱 및 API ---
app = FastAPI(title="Google Drive RAG Application", version="1.3.0-final", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, title="질문", description="RAG 모델에게 할 질문 (최소 5자 이상)")

@app.get("/")
async def root():
    return {"message": "Google Drive RAG API가 실행 중입니다."}

@app.post("/query")
async def query_documents(req: QueryRequest):
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA 체인이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.")
    try:
        question = req.question.strip()
        result = qa_chain.invoke({"query": question})
        return {"question": question, "answer": result['result']}
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
        "drive_service": "connected" if get_drive_service() else "not_connected",
        "rag_components_initialized": qa_chain is not None,
        "scheduler_running": scheduler.running if scheduler else False
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)