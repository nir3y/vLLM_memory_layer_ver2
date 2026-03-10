"""
메모리 레이어 MCP 서버 (FastMCP 방식)

Cowork(Claude)에서 이 서버를 호출하면:
- search_memory    : 관련 과거 대화 검색
- get_profile      : 5개 카테고리별 사용자 프로필 반환
- add_memory       : 새 대화 ChromaDB에 추가
- update_profiles  : 현재 대화를 분석해 프로필 MD 파일 업데이트

Claude Desktop MCP 설정 (claude_desktop_config.json):
{
  "mcpServers": {
    "memory-layer": {
      "command": "C:/Users/kwater/OneDrive/바탕 화면/vLLM_rev2/vLLM_memory_layer_Mem0-cowork/.venv/Scripts/python.exe",
      "args": ["C:/Users/kwater/OneDrive/바탕 화면/vLLM_rev2/vLLM_memory_layer_Mem0-cowork/mcp_server.py"]
    }
  }
}
"""

import sys
import os
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "mcp_server.log"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        import datetime
        f.write(f"[{datetime.datetime.now()}] {msg}\n")

try:
    log("=== 서버 시작 ===")
    sys.path.insert(0, str(BASE_DIR))
    os.chdir(BASE_DIR)
    log(f"BASE_DIR: {BASE_DIR}")

    from mcp.server.fastmcp import FastMCP
    log("FastMCP import OK")
    # memory_pipeline은 무거운 패키지라 도구 호출 시점에 lazy import

except Exception as e:
    log(f"시작 오류: {e}\n{traceback.format_exc()}")
    sys.exit(1)

mcp = FastMCP("memory-layer")

MEMORY_DIR = BASE_DIR / "memory"
DB_PATH = str(BASE_DIR / "memory_db")

PROFILE_FILES = [
    ("profile_01_identity.md",    "직업 및 전문 정체성"),
    ("profile_02_personality.md", "성격 및 소통 스타일"),
    ("profile_03_interests.md",   "관심사 및 학습 분야"),
    ("profile_04_emotional.md",   "감정 패턴 및 가치관"),
    ("profile_05_ai_usage.md",    "AI 활용 방식"),
]

CATEGORY_MAP = {
    "identity":    "profile_01_identity.md",
    "personality": "profile_02_personality.md",
    "interests":   "profile_03_interests.md",
    "emotional":   "profile_04_emotional.md",
    "ai_usage":    "profile_05_ai_usage.md",
}


# ─────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────
def _read_profile(filename: str) -> str:
    path = MEMORY_DIR / filename
    if not path.exists():
        return f"[{filename}: 아직 생성되지 않음 — generate_profile.py 먼저 실행 필요]"
    return path.read_text(encoding="utf-8")


def _update_single_profile(filename: str, label: str, conversation: str) -> str:
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model="qwen2.5:3b", num_ctx=2048)
    except Exception as e:
        return f"Ollama 연결 실패: {e}"

    existing = _read_profile(filename)
    prompt = f"""다음은 사용자의 현재 프로필과 최근 대화입니다.
최근 대화에서 '{label}' 관련 새로운 정보만 추출하여 기존 프로필에 추가/수정해주세요.

[기존 프로필]
{existing}

[최근 대화]
{conversation[:3000]}

규칙:
- 기존 내용은 최대한 유지
- 새로운 정보만 추가 또는 기존 내용 보완
- 한국어로 작성
- 기존 프로필 형식(마크다운 헤더 구조) 유지
- 명확하지 않은 것은 추가하지 않음

업데이트된 프로필:"""

    try:
        updated = llm.invoke(prompt)
        path = MEMORY_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
        return f"✅ {filename} 업데이트 완료"
    except Exception as e:
        return f"❌ {filename} 업데이트 실패: {e}"


# ─────────────────────────────────────────
# 도구 정의
# ─────────────────────────────────────────
@mcp.tool()
def search_memory_tool(query: str, k: int = 3) -> str:
    """
    과거 대화 기록에서 현재 대화와 관련된 기억을 검색합니다.
    사용자가 언급한 주제, 업무, 관심사와 관련된 과거 대화를 찾아 반환합니다.
    예: '댐 설계', '투자 관련 대화', '가족 이야기'

    Args:
        query: 검색할 주제나 키워드
        k: 반환할 결과 수 (기본값: 3)
    """
    from memory_pipeline import search_memory
    results = search_memory(query=query, db_path=DB_PATH, k=k)

    if not results:
        return "관련 기억을 찾지 못했습니다."

    output = f"🔍 '{query}' 관련 기억 {len(results)}개:\n\n"
    for i, doc in enumerate(results, 1):
        date = doc.metadata.get("date", "날짜 없음")
        topic = doc.metadata.get("topic", "")
        preview = doc.page_content[:300].replace("\n", " ")
        output += f"[{i}] 📄 {topic} ({date})\n"
        output += f"     {preview}...\n\n"

    return output


@mcp.tool()
def get_profile(category: str = "all") -> str:
    """
    사용자의 5개 카테고리별 프로필을 반환합니다.
    세션 시작 시 호출하여 사용자 맥락을 파악하세요.
    카테고리: all(기본), identity, personality, interests, emotional, ai_usage

    Args:
        category: 가져올 카테고리. 기본값은 'all' (전체 5개)
    """
    if category != "all" and category in CATEGORY_MAP:
        filename = CATEGORY_MAP[category]
        label = next(l for f, l in PROFILE_FILES if f == filename)
        content = _read_profile(filename)
        return f"## {label}\n\n{content}"

    output_parts = ["# 사용자 프로필 (5개 카테고리)\n"]
    for filename, label in PROFILE_FILES:
        content = _read_profile(filename)
        output_parts.append(f"---\n## {label}\n\n{content}\n")

    return "\n".join(output_parts)


@mcp.tool()
def add_memory(file_path: str) -> str:
    """
    새로운 대화 MD 파일을 메모리 DB에 추가합니다.
    대화 세션이 끝난 후 호출하여 기억을 업데이트합니다.

    Args:
        file_path: 추가할 대화 MD 파일의 전체 경로
    """
    if not Path(file_path).exists():
        return f"파일을 찾을 수 없습니다: {file_path}"

    from memory_pipeline import add_new_conversation
    add_new_conversation(file_path=file_path, db_path=DB_PATH)
    return f"✅ '{Path(file_path).name}' 메모리에 추가 완료"


@mcp.tool()
def update_profiles(conversation: str, categories: list[str] = None) -> str:
    """
    현재 대화 내용을 분석해 관련 프로필 MD 파일을 업데이트합니다.
    대화가 끝나거나 마무리될 때 호출하세요.

    Args:
        conversation: 분석할 대화 내용 (현재 세션 전체 또는 주요 부분)
        categories: 업데이트할 카테고리 목록. 기본값: 모든 카테고리
                    예: ['identity', 'interests']
    """
    if categories is None:
        categories = list(CATEGORY_MAP.keys())

    results = []
    for category_key in categories:
        if category_key not in CATEGORY_MAP:
            results.append(f"⚠️ 알 수 없는 카테고리: {category_key}")
            continue

        filename = CATEGORY_MAP[category_key]
        label = next(l for f, l in PROFILE_FILES if f == filename)
        result = _update_single_profile(filename, label, conversation)
        results.append(result)

    return "프로필 업데이트 완료:\n\n" + "\n".join(results)


# ─────────────────────────────────────────
# 서버 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    log("mcp.run() 호출")
    mcp.run()
    log("mcp.run() 종료")
