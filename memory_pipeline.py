"""
메모리 레이어 파이프라인
- MD 대화 파일 → (User + Assistant) 쌍 파싱 → 임베딩 → ChromaDB 저장
"""

import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ─────────────────────────────────────────
# 1단계: MD 파일 파서
# ─────────────────────────────────────────
def parse_md_file(file_path: str) -> list[Document]:
    for encoding in ["utf-8", "cp949", "euc-kr"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        # 전부 실패하면 오류 문자 무시하고 읽기
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    # 메타데이터 추출 
    filename = Path(file_path).stem  # 예: "2025-06-24_SaaS화 대체 전략"

    # 파일명 앞 날짜 파싱
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    date = date_match.group(1) if date_match else "unknown"

    # 날짜 이후 주제 파싱
    topic = filename[11:] if date_match else filename

    # 파일 내부의 ID 추출
    id_match = re.search(r"- ID: ([a-f0-9\-]+)", content)
    source_id = id_match.group(1) if id_match else filename

    # ── User + Assistant 쌍 추출 ─────────────
    # "## 👤 User" 다음에 오는 "## 🤖 Assistant" 까지를 하나의 쌍으로 묶음
    pair_pattern = r"## 👤 User\s*\n(.*?)## 🤖 Assistant\s*\n(.*?)(?=## 👤 User|## 🤖 Assistant|\Z)"
    pairs = re.findall(pair_pattern, content, re.DOTALL)

    documents = []
    for i, (user_text, assistant_text) in enumerate(pairs):
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()

        if not user_text:
            continue

        # User 질문 + Assistant 답변을 하나로 묶어서 저장
        combined = f"[User]\n{user_text}\n\n[Assistant]\n{assistant_text}"

        documents.append(
            Document(
                page_content=combined,
                metadata={
                    "date": date,
                    "topic": topic,
                    "source_id": source_id,
                    "source_file": Path(file_path).name,
                    "turn_index": i,      # 같은 파일 내 몇 번째 쌍인지
                    "user_text": user_text,  # 검색 디버깅용 (User 발화만 따로 보관)
                },
            )
        )

    return documents


# ─────────────────────────────────────────
# 2단계: 전체 빌드 (초기 1회)
# ─────────────────────────────────────────
def build_memory_db(md_folder: str, db_path: str = "./memory_db") -> Chroma:
    # 로컬 임베딩 모델 
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 모든 MD 파일 수집 및 파싱
    all_documents = []
    md_files = sorted(Path(md_folder).glob("*.md"))

    print(f"총 {len(md_files)}개 파일 발견")

    for i, md_file in enumerate(md_files):
        docs = parse_md_file(str(md_file))
        all_documents.extend(docs)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(md_files)} 파일 파싱 완료...")

    print(f"총 {len(all_documents)}개 대화 쌍 추출 → 임베딩 시작")

    # ChromaDB에 한 번에 저장 (임베딩은 내부에서 자동 처리)
    db = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="memory",
    )

    print(f"완료! '{db_path}'에 저장됨")
    return db





# ─────────────────────────────────────────
# 3단계: 새 대화 추가 (세션마다)
# ─────────────────────────────────────────
def add_new_conversation(file_path: str, db_path: str = "./memory_db"):
    """
    새 MD 파일 1개를 기존 ChromaDB에 추가.
    Cowork 세션 종료 후 자동 호출되는 함수.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="memory",
    )

    new_docs = parse_md_file(file_path)
    db.add_documents(new_docs)

    print(f"'{Path(file_path).name}' → {len(new_docs)}개 쌍 추가 완료")


# ─────────────────────────────────────────
# 4단계: 검색
# ─────────────────────────────────────────
def search_memory(
    query: str,
    db_path: str = "./memory_db",
    k: int = 5,
    date_filter: str = None,
) -> list[Document]:
    """
    저장된 메모리에서 관련 대화 쌍 검색.
    MMR 방식으로 다양한 맥락의 결과를 반환.

    Args:
        query     : 검색어 (예: "투자 관련 관심사", "수문학 업무")
        k         : 반환할 결과 수
        date_filter: "2025-06" 처럼 특정 연월로 필터링 (선택)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="memory",
    )

    search_kwargs = {"k": k, "fetch_k": k * 4, "lambda_mult": 0.5}

    # 날짜 필터 적용 (선택)
    if date_filter:
        search_kwargs["filter"] = {"date": {"$contains": date_filter}}

    retriever = db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    results = retriever.invoke(query)

    return results


# ─────────────────────────────────────────
# 직접 실행 시 ChromaDB 빌드
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    md_folder = sys.argv[1] if len(sys.argv) > 1 else "./conversations"
    db_path   = sys.argv[2] if len(sys.argv) > 2 else "./memory_db"

    print(f"MD 폴더: {md_folder}")
    print(f"DB 경로: {db_path}")

    build_memory_db(md_folder=md_folder, db_path=db_path)
