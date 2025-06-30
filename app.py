# app.py (Improved Version - 파일 타입별 최적화 및 중복 방지)

import uvicorn
import os
import logging
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import uuid
import re
from pydantic import BaseModel, Field
from typing import List
from collections import Counter
from datetime import datetime, timedelta

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
try:
    import pypdf
except ImportError:
    logging.warning("pypdf 라이브러리가 설치되지 않았습니다.")
    pypdf = None

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
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
EMBEDDING_BATCH_SIZE = 32  # 메모리 효율성을 위해 감소
CHUNK_VERSION = f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_optimized_v5"  # 새 버전

# --- 3. 전역 변수 ---
vectorstore = None
qa_chain = None
embeddings = None
scheduler = BackgroundScheduler(timezone="Asia/Seoul")
drive_service = None
sync_in_progress = False

# --- 4. Google Drive 연동 및 파일 파싱 ---
def get_drive_service():
    global drive_service
    if drive_service: 
        return drive_service
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
    """다양한 인코딩으로 바이트를 텍스트로 디코딩"""
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'iso-8859-1', 'latin1']
    for encoding in encodings:
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError: 
            continue
    logger.warning(f"파일 {file_id} 디코딩 실패. 일부 문자 대체됨.")
    return content_bytes.decode('utf-8', errors='replace')

def parse_excel_to_markdown(file_io, file_name):
    """엑셀 파일을 마크다운 테이블로 변환"""
    try:
        # 엔진 호환성 확보
        try:
            xls = pd.ExcelFile(file_io, engine='openpyxl')
        except Exception:
            try:
                xls = pd.ExcelFile(file_io, engine='xlrd')
            except Exception as e:
                logger.error(f"엑셀 파일 읽기 실패 ({file_name}): {e}")
                return None
                
        markdown_parts = []
        
        logger.info(f"📊 엑셀 파일 '{file_name}' 마크다운 변환 시작. 시트 목록: {xls.sheet_names}")
        
        # 파일 메타데이터
        markdown_parts.append(f"# 📊 {file_name}\n\n")
        markdown_parts.append(f"**파일 형식:** Microsoft Excel\n")
        markdown_parts.append(f"**시트 수:** {len(xls.sheet_names)}\n")
        markdown_parts.append(f"**시트 목록:** {', '.join(xls.sheet_names)}\n\n")
        
        processed_sheets = 0
        
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(
                    xls, 
                    sheet_name=sheet_name, 
                    header=None, 
                    dtype=str, 
                    keep_default_na=False,
                    na_filter=False
                )
                
                if df.empty or df.shape[0] == 0: 
                    logger.info(f"  - 시트 '{sheet_name}': 빈 시트")
                    continue
                
                # 시트 헤더
                markdown_parts.append(f"## 📋 시트: {sheet_name}\n\n")
                
                # 빈 행 제거
                non_empty_rows = []
                for idx, row in df.iterrows():
                    if any(str(cell).strip() and str(cell).lower() not in ['nan', 'none', 'null', ''] for cell in row):
                        non_empty_rows.append(idx)
                
                if not non_empty_rows:
                    markdown_parts.append("*데이터가 없는 시트*\n\n")
                    continue
                
                df_filtered = df.iloc[non_empty_rows].reset_index(drop=True)
                
                # 헤더 감지 (처음 5행 중 가장 많은 데이터를 가진 행)
                header_row_idx = None
                max_non_empty = 0
                
                for idx in range(min(5, len(df_filtered))):
                    row = df_filtered.iloc[idx]
                    non_empty_count = sum(
                        1 for cell in row 
                        if str(cell).strip() and str(cell).lower() not in ['nan', 'none', 'null', 'unnamed']
                    )
                    if non_empty_count > max_non_empty and non_empty_count >= 2:
                        max_non_empty = non_empty_count
                        header_row_idx = idx
                
                if header_row_idx is not None:
                    # 테이블 형식으로 변환
                    header_row = df_filtered.iloc[header_row_idx]
                    headers = []
                    
                    for i, cell in enumerate(header_row):
                        cell_str = str(cell).strip()
                        if cell_str and cell_str.lower() not in ['nan', 'none', 'null', 'unnamed']:
                            clean_header = cell_str.replace('|', '\\|').replace('\n', ' ').replace('\r', '')
                            headers.append(clean_header)
                        else:
                            headers.append(f"컬럼{i+1}")
                    
                    # 최대 컬럼 수 제한
                    max_columns = min(len(headers), 10)
                    headers = headers[:max_columns]
                    
                    # 마크다운 테이블 헤더
                    markdown_parts.append("| " + " | ".join(headers) + " |\n")
                    markdown_parts.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
                    
                    # 데이터 행들
                    data_rows_added = 0
                    max_rows = 100
                    
                    for row_idx in range(header_row_idx + 1, len(df_filtered)):
                        if data_rows_added >= max_rows:
                            markdown_parts.append(f"| ... | *({len(df_filtered) - header_row_idx - 1 - max_rows}개 행 더 있음)* | ... |\n")
                            break
                            
                        row = df_filtered.iloc[row_idx]
                        row_data = []
                        
                        for i in range(len(headers)):
                            if i < len(row):
                                cell = row.iloc[i]
                                cell_str = str(cell).strip()
                                if cell_str and cell_str.lower() not in ['nan', 'none', 'null']:
                                    clean_cell = (cell_str.replace('|', '\\|')
                                                        .replace('\n', '<br>')
                                                        .replace('\r', ''))
                                    if len(clean_cell) > 100:
                                        clean_cell = clean_cell[:97] + "..."
                                    row_data.append(clean_cell)
                                else:
                                    row_data.append("")
                            else:
                                row_data.append("")
                        
                        if any(cell.strip() for cell in row_data):
                            markdown_parts.append("| " + " | ".join(row_data) + " |\n")
                            data_rows_added += 1
                
                else:
                    # 헤더가 없는 경우 리스트 형식
                    markdown_parts.append("### 📝 데이터 (구조화되지 않은 형식)\n\n")
                    
                    rows_added = 0
                    max_list_rows = 50
                    
                    for row_idx, row in df_filtered.iterrows():
                        if rows_added >= max_list_rows:
                            markdown_parts.append(f"*...및 {len(df_filtered) - max_list_rows}개 행 더*\n\n")
                            break
                            
                        row_items = []
                        for cell in row:
                            cell_str = str(cell).strip()
                            if cell_str and cell_str.lower() not in ['nan', 'none', 'null']:
                                if len(cell_str) > 50:
                                    cell_str = cell_str[:47] + "..."
                                row_items.append(cell_str)
                        
                        if row_items:
                            markdown_parts.append(f"- **행 {rows_added + 1}:** {' | '.join(row_items)}\n")
                            rows_added += 1
                
                markdown_parts.append("\n")
                processed_sheets += 1
                logger.info(f"  ✅ 시트 '{sheet_name}' 처리 완료")
                
            except Exception as sheet_error:
                logger.error(f"시트 '{sheet_name}' 처리 중 오류: {sheet_error}")
                markdown_parts.append(f"*❌ 시트 '{sheet_name}' 처리 중 오류 발생: {str(sheet_error)[:100]}*\n\n")
                continue
        
        if processed_sheets == 0:
            logger.warning(f"엑셀 파일 '{file_name}'에서 처리 가능한 시트가 없음")
            return None
            
        final_content = "".join(markdown_parts)
        logger.info(f"📊 엑셀 변환 완료: {processed_sheets}개 시트, {len(final_content)}자")
        return final_content
        
    except Exception as e:
        logger.error(f"엑셀 마크다운 변환 오류 ({file_name}): {e}", exc_info=True)
        return None

def parse_pdf_to_text(file_io, file_name):
    """PDF를 순수 텍스트로 변환 (마크다운 최소화)"""
    if not pypdf:
        logger.error("PDF 처리를 위해 pypdf 라이브러리가 필요합니다.")
        return None
    
    try:
        pdf_reader = pypdf.PdfReader(file_io)
        parts = [f"문서명: {file_name}\n"]
        
        logger.info(f"📄 PDF 파일 '{file_name}' 처리 시작. 총 {len(pdf_reader.pages)}페이지")
        
        # PDF 메타데이터 (간단하게)
        if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
            doc_meta = pdf_reader.metadata
            if doc_meta.get('/Title'):
                parts.append(f"제목: {doc_meta['/Title']}\n")
            if doc_meta.get('/Author'):
                parts.append(f"작성자: {doc_meta['/Author']}\n")
            if doc_meta.get('/Subject'):
                parts.append(f"주제: {doc_meta['/Subject']}\n")
        
        parts.append(f"총 페이지: {len(pdf_reader.pages)}\n\n")
        
        # 페이지별 내용 (간단한 구분자만 사용)
        for i, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    parts.append(f"=== 페이지 {i+1} ===\n")
                    cleaned_text = re.sub(r'\n\s*\n', '\n\n', text.strip())
                    cleaned_text = re.sub(r' +', ' ', cleaned_text)
                    parts.append(cleaned_text + "\n\n")
            except Exception as page_err:
                logger.warning(f"PDF 페이지 {i+1} 처리 중 오류: {page_err}")
                continue
        
        return "".join(parts)
        
    except Exception as e:
        logger.error(f"PDF 파일 파싱 오류 ({file_name}): {e}")
        return None

def parse_csv_to_text(file_io, file_name):
    """CSV를 검색 친화적인 텍스트로 변환"""
    try:
        content = _decode_bytes_to_text(file_io.getvalue(), "")
        lines = content.strip().split('\n')
        
        if not lines:
            return None
            
        parts = [f"파일명: {file_name}\n"]
        parts.append(f"데이터 형식: CSV\n")
        parts.append(f"총 행수: {len(lines)}\n\n")
        
        # 헤더 (첫 번째 행)
        if lines:
            headers = [h.strip().strip('"') for h in lines[0].split(',')]
            parts.append(f"컬럼: {', '.join(headers)}\n\n")
        
        # 데이터 샘플 (최대 20행)
        parts.append("데이터 샘플:\n")
        for i, line in enumerate(lines[:min(20, len(lines))]):
            if line.strip():
                # CSV 특수문자 정리
                clean_line = line.replace('"', '').strip()
                parts.append(f"행{i+1}: {clean_line}\n")
        
        if len(lines) > 20:
            parts.append(f"... 및 {len(lines) - 20}개 행 더\n")
            
        return "".join(parts)
        
    except Exception as e:
        logger.error(f"CSV 파싱 오류 ({file_name}): {e}")
        return None

def parse_json_to_text(file_io, file_name):
    """JSON을 검색 친화적인 텍스트로 변환"""
    try:
        content = _decode_bytes_to_text(file_io.getvalue(), "")
        data = json.loads(content)
        
        parts = [f"파일명: {file_name}\n"]
        parts.append(f"데이터 형식: JSON\n\n")
        
        # JSON 구조 간단 분석
        def analyze_json_structure(obj, prefix="", depth=0):
            if depth > 3:  # 깊이 제한
                return []
            result = []
            if isinstance(obj, dict):
                for key, value in list(obj.items())[:10]:  # 최대 10개 키만
                    current_path = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (dict, list)):
                        result.append(f"{current_path}: {type(value).__name__}")
                        if depth < 2:  # 깊이 2까지만
                            result.extend(analyze_json_structure(value, current_path, depth + 1))
                    else:
                        value_str = str(value)[:100]
                        result.append(f"{current_path}: {value_str}")
            elif isinstance(obj, list) and obj:
                result.append(f"{prefix}[배열 {len(obj)}개 항목]")
                if len(obj) > 0 and depth < 2:
                    result.extend(analyze_json_structure(obj[0], f"{prefix}[0]", depth + 1))
            return result[:30]  # 최대 30개 항목만
        
        structure_info = analyze_json_structure(data)
        parts.append("JSON 구조:\n")
        for info in structure_info:
            parts.append(f"- {info}\n")
            
        return "".join(parts)
        
    except Exception as e:
        logger.error(f"JSON 파싱 오류 ({file_name}): {e}")
        return None

def parse_file_content(file_io, file_name, mime_type, file_id):
    """파일 타입별 최적화된 콘텐츠 파싱"""
    
    # 1. 엑셀 파일만 마크다운 테이블로 변환
    excel_mime_types = [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
        'application/vnd.ms-excel',  # .xls
        'application/vnd.ms-excel.sheet.macroEnabled.12',  # .xlsm
        'application/vnd.ms-excel.sheet.binary.macroEnabled.12'  # .xlsb
    ]
    
    if mime_type in excel_mime_types:
        logger.info(f"📊 엑셀 파일 → 마크다운 테이블 변환: {file_name}")
        return parse_excel_to_markdown(file_io, file_name)

    # 2. PDF 파일 → 순수 텍스트
    elif mime_type == 'application/pdf':
        logger.info(f"📄 PDF 파일 → 텍스트 변환: {file_name}")
        return parse_pdf_to_text(file_io, file_name)

    # 3. CSV 파일 → 구조화된 텍스트
    elif mime_type == 'text/csv':
        logger.info(f"📊 CSV 파일 → 구조화된 텍스트: {file_name}")
        return parse_csv_to_text(file_io, file_name)
    
    # 4. JSON 파일 → 구조화된 텍스트
    elif mime_type == 'application/json':
        logger.info(f"🔧 JSON 파일 → 구조화된 텍스트: {file_name}")
        return parse_json_to_text(file_io, file_name)

    # 5. 마크다운 파일 → 원본 보존
    elif mime_type == 'text/markdown' or file_name.lower().endswith('.md'):
        logger.info(f"📝 마크다운 파일 → 원본 보존: {file_name}")
        content = _decode_bytes_to_text(file_io.getvalue(), file_id)
        return content

    # 6. 일반 텍스트 → 최소한의 메타데이터만 추가
    elif mime_type.startswith('text/'):
        logger.info(f"📝 텍스트 파일 → 순수 텍스트: {file_name}")
        content = _decode_bytes_to_text(file_io.getvalue(), file_id)
        if not content.strip():
            return None
        return f"파일명: {file_name}\n파일형식: 텍스트\n\n{content}"

    # 7. 기타 지원하지 않는 형식
    else:
        logger.warning(f"❌ 지원하지 않는 MIME 타입: {mime_type} ({file_name})")
        return None

def download_and_parse_file(service, file_id, file_name, mime_type):
    """파일 다운로드 및 파싱"""
    request = None
    
    # Google Docs는 텍스트로 내보내기
    if mime_type == 'application/vnd.google-apps.document':
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
    elif mime_type in [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel', 
        'application/vnd.ms-excel.sheet.macroEnabled.12',
        'application/vnd.ms-excel.sheet.binary.macroEnabled.12',
        'application/pdf',
        'text/csv',
        'application/json'
    ] or mime_type.startswith('text/'):
        request = service.files().get_media(fileId=file_id)
    else:
        logger.warning(f"지원하지 않는 파일 형식: {mime_type} ({file_name})")
        return None
        
    try:
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done: 
            _, done = downloader.next_chunk()
        file_io.seek(0)
        
        # Google Docs는 특별 처리
        if mime_type == 'application/vnd.google-apps.document':
            content = _decode_bytes_to_text(file_io.getvalue(), file_id)
            return f"파일명: {file_name}\n파일형식: Google 문서\n\n{content}"
        else:
            return parse_file_content(file_io, file_name, mime_type, file_id)
            
    except Exception as e:
        logger.error(f"파일 다운로드 또는 파싱 실패 ({file_name}): {e}")
        return None

# --- 5. 개선된 동기화 로직 ---
def perform_sync():
    """완전히 개선된 동기화 함수 - 중복 업로드 완전 방지"""
    global embeddings, vectorstore, sync_in_progress
    
    if sync_in_progress:
        logger.info("동기화가 이미 실행 중입니다. 스킵합니다.")
        return
    
    sync_in_progress = True
    try:
        logger.info(f"🚀 동기화 시작")
        logger.info(f"GOOGLE_DRIVE_FOLDER_ID: {GOOGLE_DRIVE_FOLDER_ID}")
        
        if not all([embeddings, vectorstore]) or not GOOGLE_DRIVE_FOLDER_ID:
            logger.warning("RAG 컴포넌트 미초기화 또는 폴더 ID 누락으로 동기화를 건너뜁니다.")
            return

        service = get_drive_service()
        if not service: 
            return

        # Google Drive에서 파일 메타데이터 가져오기
        try:
            query = f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed=false"
            drive_files_meta = service.files().list(
                q=query, fields="files(id, name, modifiedTime, mimeType)", pageSize=1000
            ).execute().get('files', [])
            logger.info(f"Google Drive에서 {len(drive_files_meta)}개 파일 메타데이터 확인")
        except HttpError as e:
            logger.error(f"Google Drive API 오류: {e}")
            return

        qdrant_client = vectorstore.client
        processed_files, skipped_files, failed_files = 0, 0, 0

        for file_meta in drive_files_meta:
            file_id, file_name, modified_time = file_meta['id'], file_meta['name'], file_meta['modifiedTime']
            
            # ✅ LangChain + Qdrant 구조에 맞춘 중복 체크
            try:
                existing, _ = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME, 
                    limit=1, 
                    with_vectors=False,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.file_id",  # ✅ LangChain은 metadata. 접두사 필요
                                match=models.MatchValue(value=file_id)
                            )
                        ]
                    ),
                    with_payload=True
                )
                stored_payload = existing[0].payload if existing else None
            except Exception as filter_error:
                logger.warning(f"파일 {file_id} 필터링 중 오류: {filter_error}. 신규 처리로 진행.")
                stored_payload = None
            
            logger.info(f"📄 파일 '{file_name}' 분석:")
            
            should_process = True
            
            if stored_payload:
                # ✅ LangChain 구조에 맞춘 메타데이터 접근
                stored_metadata = stored_payload.get('metadata', {})
                stored_modified_time = stored_metadata.get('modified_time')
                stored_chunk_version = stored_metadata.get('chunk_version')
                
                logger.info(f"  - Qdrant 저장된 수정시간: {stored_modified_time}")
                logger.info(f"  - Drive 수정시간: {modified_time}")
                logger.info(f"  - Qdrant 저장된 버전: {stored_chunk_version}")
                logger.info(f"  - 현재 버전: {CHUNK_VERSION}")
                
                time_match = stored_modified_time == modified_time
                version_match = stored_chunk_version == CHUNK_VERSION
                
                logger.info(f"  ⏰ 수정시간 일치: {time_match}")
                logger.info(f"  🏷️ 버전 일치: {version_match}")
                
                if time_match and version_match:
                    should_process = False
                    skipped_files += 1
                    logger.info(f"  ✅ 스킵: 이미 최신 상태")
            else:
                logger.info(f"  - Qdrant에 기존 데이터 없음")
            
            if not should_process:
                continue
                
            # 파일 다운로드 및 파싱
            status = "신규" if not stored_payload else "변경됨"
            logger.info(f"  🔄 처리 필요: '{file_name}' (이유: {status})")
            logger.info(f"  🔄 파일 변환 시작...")
            
            content = download_and_parse_file(service, file_id, file_name, file_meta['mimeType'])
            
            if not content or not content.strip():
                logger.warning(f"  ⚠️ 파싱 실패 또는 빈 콘텐츠")
                failed_files += 1
                continue
            
            # ✅ LangChain 구조에 맞춘 기존 데이터 삭제
            if stored_payload:
                try:
                    delete_result = qdrant_client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="metadata.file_id",  # ✅ LangChain은 metadata. 접두사 필요
                                        match=models.MatchValue(value=file_id)
                                    )
                                ]
                            )
                        )
                    )
                    logger.info(f"  🗑️ 기존 데이터 삭제 완료: {delete_result}")
                except Exception as delete_error:
                    logger.error(f"  ❌ 기존 데이터 삭제 실패: {delete_error}")
                    failed_files += 1
                    continue
            
            # 청킹 및 저장
            try:
                # 파일 타입에 따른 청킹 전략
                if file_meta['mimeType'] in [
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'application/vnd.ms-excel'
                ]:
                    # 엑셀의 경우 마크다운 구조를 고려한 분리자 사용
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE, 
                        chunk_overlap=CHUNK_OVERLAP,
                        separators=["\n\n", "\n", "| ", "|", ".", " ", ""]
                    )
                else:
                    # 일반 텍스트의 경우 표준 분리자 사용
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE, 
                        chunk_overlap=CHUNK_OVERLAP,
                        separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
                    )
                
                chunks = splitter.split_text(content)
                
                if not chunks:
                    logger.warning(f"  ⚠️ 청킹 결과가 비어있음")
                    failed_files += 1
                    continue
                
                # ✅ 메타데이터를 Document의 metadata에 직접 저장
                docs_to_embed = [
                    Document(
                        page_content=chunk, 
                        metadata={
                            "source": file_name,
                            "file_id": file_id,
                            "modified_time": modified_time,
                            "chunk_version": CHUNK_VERSION,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "mime_type": file_meta['mimeType'],
                            "file_type": get_file_type_category(file_meta['mimeType'])
                        }
                    ) for i, chunk in enumerate(chunks) if chunk.strip()
                ]
                
                if not docs_to_embed:
                    logger.warning(f"  ⚠️ 유효한 청크가 없음")
                    failed_files += 1
                    continue
                
                # ✅ 배치 단위로 안전하게 저장
                for i in range(0, len(docs_to_embed), EMBEDDING_BATCH_SIZE):
                    batch = docs_to_embed[i:i + EMBEDDING_BATCH_SIZE]
                    try:
                        vectorstore.add_documents(batch)
                        logger.info(f"    📦 배치 {i//EMBEDDING_BATCH_SIZE + 1}/{(len(docs_to_embed)-1)//EMBEDDING_BATCH_SIZE + 1} 저장 완료 ({len(batch)}개 청크)")
                    except Exception as batch_error:
                        logger.error(f"    ❌ 배치 저장 실패: {batch_error}")
                        raise
                
                avg_chunk_size = sum(len(doc.page_content) for doc in docs_to_embed) // len(docs_to_embed)
                logger.info(f"  ✅ 처리 완료: {len(docs_to_embed)}개 청크 생성")
                logger.info(f"     📊 평균 청크 크기: {avg_chunk_size}자")
                
                processed_files += 1
                
            except Exception as e:
                logger.error(f"  ❌ 벡터 저장 실패: {e}", exc_info=True)
                failed_files += 1
        
        logger.info(f"🎉 동기화 완료: 처리 {processed_files}개, 건너뜀 {skipped_files}개, 실패 {failed_files}개")
    
    except Exception as e:
        logger.error(f"💥 동기화 중 예상치 못한 오류: {e}", exc_info=True)
    finally:
        sync_in_progress = False

def get_file_type_category(mime_type):
    """파일 타입 카테고리 분류"""
    if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        return "excel"
    elif mime_type == 'application/pdf':
        return "pdf"
    elif mime_type == 'text/csv':
        return "csv"
    elif mime_type == 'application/json':
        return "json"
    elif mime_type == 'application/vnd.google-apps.document':
        return "google_doc"
    elif mime_type.startswith('text/'):
        return "text"
    else:
        return "other"
# --- 6. FastAPI 수명주기 (수정된 버전) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, qa_chain, embeddings, scheduler
    logger.info("애플리케이션 시작...")

    # --- 1. RAG 컴포넌트 초기화 ---
    try:
        # LangChainDeprecationWarning 해결: 최신 라이브러리 사용
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        # 임베딩 모델 테스트 및 벡터 차원 확인
        test_embedding = embeddings.embed_query("test")
        vector_size = len(test_embedding)
        logger.info(f"Ollama 임베딩 모델({EMBEDDING_MODEL}) 로드 완료. 벡터 차원: {vector_size}")
    except Exception as e:
        logger.critical(f"임베딩 모델 초기화 실패: {e}", exc_info=True)
        raise RuntimeError("Embedding model initialization failed") from e

    # --- 2. Qdrant 클라이언트 및 컬렉션 설정 (가장 중요한 수정 부분) ---
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # 컬렉션 존재 여부를 더 안정적인 방식으로 확인
        collections_response = client.get_collections()
        existing_collections = {c.name for c in collections_response.collections}
        collection_exists = COLLECTION_NAME in existing_collections

        if collection_exists:
            info = client.get_collection(collection_name=COLLECTION_NAME)
            current_vector_size = info.config.params.vectors.size
            logger.info(f"📊 기존 컬렉션 '{COLLECTION_NAME}' 발견 - DB 벡터 차원: {current_vector_size}, 현재 모델 차원: {vector_size}")

            # 벡터 차원이 다른 경우에만 재생성 (환경 변수 체크 포함)
            force_skip = os.getenv("FORCE_SKIP_RECREATE", "false").lower() == "true"
            if current_vector_size != vector_size and not force_skip:
                logger.warning(f"⚠️ 벡터 차원 불일치! 컬렉션을 재생성합니다. (기존 데이터 삭제됨)")
                client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                )
            elif force_skip and current_vector_size != vector_size:
                 logger.warning(f"🔒 강제 보존 모드: 벡터 차원이 다르지만 기존 컬렉션을 유지합니다.")
            else:
                logger.info("✅ 기존 컬렉션 재사용.")
        else:
            # 컬렉션이 존재하지 않을 때만 새로 생성 (데이터 보존)
            logger.info(f"📝 새 컬렉션 '{COLLECTION_NAME}' 생성")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )

    except Exception as e:
        logger.critical(f"💥 Qdrant 컬렉션 초기화 중 심각한 오류 발생: {e}", exc_info=True)
        raise RuntimeError("Failed to initialize Qdrant collection") from e

    # LangChainDeprecationWarning 해결: 최신 라이브러리 사용
    vectorstore = Qdrant(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embeddings=embeddings
    )
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    
    # --- 3. QA 체인 및 Retriever 설정 ---
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={'k': 8}
    )
    
    improved_prompt = PromptTemplate(
        template="""당신은 업무 문서를 분석하여 한국어로 답변하는 전문가입니다.

**답변 규칙:**
1. 제공된 문서 내용에서만 정보를 찾아 답변하세요
2. 문서에 없는 정보는 "문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요
3. 답변은 한국어로 명확하고 간결하게 작성하세요
4. 표나 데이터가 포함된 경우 구체적인 수치나 항목을 포함하여 답변하세요

**문서 내용:**
{context}

**질문:** {question}

**답변:**""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": improved_prompt},
        return_source_documents=True
    )
    logger.info("🎯 RAG 컴포넌트 초기화 완료.")

    # --- 4. 스케줄러 설정 (수정된 부분) ---
    if not scheduler.running:
        # 단일 작업으로 통합하고, 첫 실행에 10초 지연을 줌
        scheduler.add_job(
            perform_sync,
            'interval',
            minutes=SYNC_INTERVAL_MINUTES,
            id="gdrive_sync_job",  # 단일 ID 사용
            next_run_time=datetime.now() + timedelta(seconds=10) # 재시작 시 안정성을 위한 지연
        )
        scheduler.start()
        logger.info(f"자동 동기화 스케줄러 시작. 10초 후 첫 동기화를 시작하며, 이후 {SYNC_INTERVAL_MINUTES}분마다 실행됩니다.")

    yield

    logger.info("애플리케이션 종료...")
    if scheduler.running: 
        scheduler.shutdown()
        
# --- 7. FastAPI 앱 및 API ---
app = FastAPI(title="Optimized Google Drive RAG Application", version="6.0.0-optimized", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, title="질문", description="RAG 모델에게 할 질문")

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources_used: int = None
    file_types_used: List[str] = []

@app.get("/")
async def root():
    return {"message": "Optimized Google Drive RAG API가 실행 중입니다. /docs 에서 API 문서를 확인하세요."}

@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """개선된 쿼리 처리"""
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA 체인이 아직 초기화되지 않았습니다.")
    
    try:
        question = req.question.strip()
        logger.info(f"🔍 질문: {question}")
        
        result = qa_chain.invoke({"query": question})
        
        # 소스 문서 정보 수집
        source_docs = result.get('source_documents', [])
        sources_used = len(source_docs)
        
        # 사용된 파일 타입 분석
        file_types_used = list(set(
            doc.metadata.get('file_type', 'unknown') 
            for doc in source_docs 
            if doc.metadata.get('file_type')
        ))
        
        return {
            "question": question,
            "answer": result['result'].strip(),
            "sources_used": sources_used,
            "file_types_used": file_types_used
        }
        
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="질문 처리 중 서버 오류가 발생했습니다.")

@app.post("/manual-sync", status_code=202)
async def manual_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(perform_sync)
    return {"message": "수동 동기화 작업이 시작되었습니다."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "6.0.0-optimized",
        "drive_service": "connected" if drive_service else "not_connected",
        "rag_components_initialized": qa_chain is not None,
        "scheduler_running": scheduler.running if scheduler else False,
        "chunk_version": CHUNK_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "sync_in_progress": sync_in_progress
    }

@app.get("/debug/search")
async def debug_search(query: str, limit: int = 5):
    """검색 결과 디버깅용 엔드포인트"""
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        docs = vectorstore.similarity_search(query, k=limit)
        
        return {
            "query": query,
            "total_results": len(docs),
            "results": [
                {
                    "content": doc.page_content[:300] + "...",
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content),
                    "file_type": doc.metadata.get('file_type', 'unknown')
                }
                for doc in docs
            ]
        }
    except Exception as e:
        logger.error(f"디버그 검색 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/collection-info")
async def collection_info():
    """수정된 Qdrant 컬렉션 정보 확인"""
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        client = vectorstore.client
        info = client.get_collection(COLLECTION_NAME)
        
        # 파일별 통계 계산
        file_stats = {}
        file_type_stats = {}
        
        all_points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        for point in all_points:
            file_id = point.payload.get('metadata', {}).get('file_id', 'unknown')
            source = point.payload.get('metadata', {}).get('source', 'unknown')
            file_type = point.payload.get('metadata', {}).get('file_type', 'unknown')
            
            # 파일별 통계
            if file_id not in file_stats:
                metadata = point.payload.get('metadata', {})
                file_stats[file_id] = {
                    'source': source,
                    'chunk_count': 0,
                    'modified_time': metadata.get('modified_time'),
                    'chunk_version': metadata.get('chunk_version'),
                    'file_type': file_type
                }
            file_stats[file_id]['chunk_count'] += 1
            
            # 파일 타입별 통계
            if file_type not in file_type_stats:
                file_type_stats[file_type] = {'count': 0, 'files': 0}
            file_type_stats[file_type]['count'] += 1
            if file_id not in [f['file_id'] for f in file_type_stats[file_type].get('file_list', [])]:
                file_type_stats[file_type]['files'] += 1
        
        # 샘플 데이터 조회
        sample_points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        return {
            "collection_name": COLLECTION_NAME,
            "total_points": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance_metric": info.config.params.vectors.distance.value,
            "total_files": len(file_stats),
            "file_type_statistics": file_type_stats,
            "file_statistics": file_stats,
            "sample_points": [
                {
                    "id": point.id,
                    "file_id": point.payload.get('metadata', {}).get('file_id', 'unknown'),
                    "source": point.payload.get('metadata', {}).get('source', 'unknown'),
                    "file_type": point.payload.get('metadata', {}).get('file_type', 'unknown'),
                    "modified_time": point.payload.get('metadata', {}).get('modified_time'),
                    "chunk_version": point.payload.get('metadata', {}).get('chunk_version'),
                    "chunk_index": point.payload.get('metadata', {}).get('chunk_index'),
                    "content_preview": point.payload.get('page_content', '')[:200] + "..."
                }
                for point in sample_points
            ]
        }
    except Exception as e:
        logger.error(f"컬렉션 정보 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/file-check/{file_id}")
async def check_file_exists(file_id: str):
    """특정 파일의 존재 여부 확인"""
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        client = vectorstore.client
        
        # ✅ LangChain 구조에 맞춘 검색
        existing, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.file_id",  # ✅ LangChain은 metadata. 접두사 필요
                        match=models.MatchValue(value=file_id)
                    )
                ]
            )
        )
        
        if existing:
            chunks_info = []
            for point in existing:
                chunks_info.append({
                    "chunk_index": point.payload.get('chunk_index'),
                    "content_length": len(point.payload.get('page_content', '')),
                    "content_preview": point.payload.get('page_content', '')[:100] + "..."
                })
            
            return {
                "file_id": file_id,
                "exists": True,
                "total_chunks": len(existing),
                "file_info": {
                    "source": existing[0].payload.get('metadata', {}).get('source'),
                    "modified_time": existing[0].payload.get('metadata', {}).get('modified_time'),
                    "chunk_version": existing[0].payload.get('metadata', {}).get('chunk_version'),
                    "file_type": existing[0].payload.get('metadata', {}).get('file_type')
                },
                "chunks": chunks_info
            }
        else:
            return {
                "file_id": file_id,
                "exists": False,
                "total_chunks": 0
            }
            
    except Exception as e:
        logger.error(f"파일 존재 확인 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)