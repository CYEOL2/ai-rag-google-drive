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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
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
    """바이트를 텍스트로 디코딩 (한글 지원 강화)"""
    # 다양한 인코딩 시도 (한글 지원을 위해 순서 중요)
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'iso-8859-1', 'latin1']
    
    for encoding in encodings:
        try: 
            decoded = content_bytes.decode(encoding)
            logger.info(f"파일 {file_id} 디코딩 성공: {encoding}")
            return decoded
        except UnicodeDecodeError: 
            continue
    
    # 모든 인코딩 실패시 오류 복구 시도
    try:
        decoded = content_bytes.decode('utf-8', errors='replace')
        logger.warning(f"파일 {file_id} 디코딩 시 일부 문자 대체됨")
        return decoded
    except Exception as e:
        logger.error(f"파일 인코딩을 해석할 수 없습니다: {file_id}, 오류: {e}")
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
            try:
                xls = pd.ExcelFile(file_io, engine='openpyxl')
                parts = []
                logger.info(f"엑셀 파일 '{file_name}' 처리 시작. 시트 목록: {xls.sheet_names}")
                
                for sheet in xls.sheet_names:
                    # dtype=str로 모든 값을 문자열로 읽고, keep_default_na=False로 NaN 방지
                    df = pd.read_excel(xls, sheet_name=sheet, dtype=str, keep_default_na=False, na_filter=False)
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if df.empty: 
                        continue
                    
                    # 컴럼명 인코딩 문제 해결
                    df.columns = [fix_filename_encoding(str(col)) for col in df.columns]
                    
                    header = f"=== 시트: {fix_filename_encoding(sheet)} ===\n"
                    
                    # 각 행을 더 상세하게 처리
                    rows_content = []
                    for idx, row in df.iterrows():
                        # 빈 값이 아닌 셀들만 추출 (공백, nan, None 제외)
                        non_empty_items = []
                        for col, val in row.items():
                            if val and str(val).strip() and str(val).lower() not in ['nan', 'none', '']:
                                # 값 인코딩 문제 해결
                                fixed_val = fix_filename_encoding(str(val))
                                non_empty_items.append(f"{col}: {fixed_val}")
                        
                        if non_empty_items:
                            row_text = " | ".join(non_empty_items)
                            rows_content.append(row_text)
                    
                    if rows_content:
                        sheet_content = header + "\n".join(rows_content)
                        parts.append(sheet_content)
                        logger.info(f"시트 '{sheet}' 처리 완료: {len(rows_content)}행")
                
                result = "\n\n".join(parts) if parts else None
                if result:
                    logger.info(f"엑셀 파일 '{file_name}' 처리 완료. 총 {len(parts)}개 시트, 길이: {len(result)}자")
                return result
                
            except Exception as excel_error:
                logger.error(f"엑셀 파일 처리 중 오류 ({file_name}): {excel_error}")
                return None

        return _decode_bytes_to_text(file_io.getvalue(), file_id)
    except Exception as e:
        logger.error(f"파일 다운로드 또는 파싱 실패 ({file_name}): {e}")
        return None

def fix_filename_encoding(file_name):
    """파일말 인코딩 문제 해결 (강화버전)"""
    if not file_name:
        return file_name
        
    try:
        # 1. 이미 올바른 UTF-8 문자열인지 확인
        try:
            # 한글 문자가 정상적으로 포함되어 있는지 확인
            if any('가' <= char <= '힣' for char in file_name):
                logger.info(f"정상 한글 파일명: {file_name}")
                return file_name
        except:
            pass
            
        # 2. 깨진 문자 패턴 감지 및 복구
        broken_patterns = ['ë', 'ì', 'ê', 'í', 'î', 'ï']
        if any(pattern in file_name for pattern in broken_patterns):
            logger.warning(f"깨진 문자 감지: {file_name}")
            
            # 다양한 인코딩 복구 시도
            recovery_attempts = [
                # Latin-1 -> UTF-8
                lambda x: x.encode('latin-1').decode('utf-8'),
                # CP1252 -> UTF-8  
                lambda x: x.encode('cp1252').decode('utf-8'),
                # ISO-8859-1 -> UTF-8
                lambda x: x.encode('iso-8859-1').decode('utf-8'),
                # Latin-1 -> CP949
                lambda x: x.encode('latin-1').decode('cp949'),
                # Latin-1 -> EUC-KR
                lambda x: x.encode('latin-1').decode('euc-kr')
            ]
            
            for i, attempt in enumerate(recovery_attempts):
                try:
                    recovered = attempt(file_name)
                    # 복구된 문자열에 한글이 있는지 확인
                    if any('가' <= char <= '힣' for char in recovered):
                        logger.info(f"파일명 복구 성공 (method {i+1}): {file_name} -> {recovered}")
                        return recovered
                except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
                    continue
        
        # 3. 복구 실패 시 원본 반환
        logger.info(f"인코딩 복구 불필요/실패: {file_name}")
        return file_name
        
    except Exception as e:
        logger.error(f"파일명 인코딩 복구 중 오류: {e}")
        return file_name

def get_files_from_drive(folder_id):
    service = get_drive_service()
    if not service: return []
    try:
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name, modifiedTime, mimeType)", pageSize=1000).execute()
        docs = []
        for file in results.get('files', []):
            file_name = file['name']
            # 파일명 인코딩 문제 해결
            file_name = fix_filename_encoding(file_name)
            
            logger.info(f"처리할 파일: {file_name} (ID: {file['id']})")
            
            if content := download_file_content(service, file['id'], file_name, file['mimeType']):
                docs.append({
                    'id': file['id'], 
                    'name': file_name, 
                    'content': content, 
                    'modified_time': file['modifiedTime']
                })
                logger.info(f"파일 로드 완료: {file_name}")
            else:
                logger.warning(f"파일 로드 실패: {file_name}")
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
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={
            'k': 10,
            'fetch_k': 20,
            'lambda_mult': 0.7
        }
    )
    # 한국어 응답을 위한 커스텀 체인 설정
    from langchain.prompts import PromptTemplate
    
    custom_prompt = PromptTemplate(
        template="""[지침사항]
- 반드시 한국어로 답변하세요
- 제공된 문서에서 정보를 찾아 정확하게 답변하세요
- 정보가 없으면 "문서에서 관련 정보를 찾을 수 없습니다"라고 한국어로 답변하세요

[문서 내용]
{context}

[질문]
{question}

[답변]""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )
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

@app.get("/debug/documents")
async def debug_documents():
    """저장된 문서들의 정보를 확인"""
    if not vectorstore:
        return {"error": "Vectorstore not initialized"}
    
    try:
        client = vectorstore.client
        # 모든 문서 정보 가져오기
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=50,
            with_payload=True,
            with_vectors=False
        )
        
        documents = []
        for point in points:
            payload = point.payload
            documents.append({
                "source": payload.get("source", "Unknown"),
                "file_id": payload.get("file_id", "Unknown"),
                "content_preview": payload.get("page_content", "")[:200] + "..."
            })
        
        return {
            "total_documents": len(points),
            "documents": documents
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug/search")
async def debug_search(req: QueryRequest):
    """검색 결과를 디버그"""
    if not vectorstore:
        return {"error": "Vectorstore not initialized"}
    
    try:
        # 직접 벡터 검색 실행
        docs = vectorstore.similarity_search_with_score(req.question, k=10)
        
        results = []
        for doc, score in docs:
            results.append({
                "score": float(score),
                "source": doc.metadata.get("source", "Unknown"),
                "content_preview": doc.page_content[:300] + "..."
            })
        
        return {
            "question": req.question,
            "search_results": results
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)