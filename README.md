# vLLM Memory Layer

Claude Desktop을 위한 **영구 메모리 레이어** MCP 서버입니다.
과거 대화를 벡터 DB에 저장하고, 세션 간 사용자 맥락을 자동으로 유지합니다.

---

## 개요

Claude는 기본적으로 대화 세션이 끝나면 이전 맥락을 기억하지 못합니다.
이 프로젝트는 **MCP(Model Context Protocol) 서버**로 동작하며, Claude Desktop에 다음 기능을 제공합니다:

- 과거 대화를 의미 기반으로 검색
- 날짜별 대화 기록 조회
- 사용자 프로필 5개 카테고리 자동 구축 및 갱신
- 새로운 대화를 메모리 DB에 실시간 추가

---

## 아키텍처

```
conversations/ (마크다운 대화 파일)
        ↓  parse_md_file()
  User/Assistant 쌍 추출
        ↓  E5Embeddings (multilingual-e5-base)
     벡터 임베딩 생성
        ↓
   ChromaDB (memory_db/)
        ↓
  FastMCP 서버 (mcp_server.py)
        ↓
   Claude Desktop
```

---

## 제공 도구 (MCP Tools)

| 도구 | 설명 |
|------|------|
| `search_memory_tool` | 쿼리 + 선택적 날짜 필터로 관련 과거 대화 검색 (MMR 방식) |
| `get_profile` | 5개 카테고리 사용자 프로필 반환 (세션 시작 시 호출) |
| `get_conversation_by_date` | 특정 날짜/월/연도의 대화 목록 조회 |
| `add_memory` | 새 대화 MD 파일을 ChromaDB에 추가 |
| `update_profiles` | 대화 내용을 분석해 프로필 MD 파일을 LLM으로 갱신 |

---

## 사용자 프로필 구조

`memory/` 폴더에 5개의 마크다운 파일로 관리됩니다:

| 파일 | 카테고리 |
|------|---------|
| `profile_01_identity.md` | 직업 및 전문 정체성 |
| `profile_02_personality.md` | 성격 및 소통 스타일 |
| `profile_03_interests.md` | 관심사 및 학습 분야 |
| `profile_04_emotional.md` | 감정 패턴 및 가치관 |
| `profile_05_ai_usage.md` | AI 활용 방식 |

---

## 기술 스택

| 컴포넌트 | 기술 |
|---------|------|
| 임베딩 모델 | `intfloat/multilingual-e5-base` (768차원, 한/영 지원) |
| 벡터 DB | ChromaDB (로컬 영구 저장) |
| 검색 방식 | MMR (Max Marginal Relevance) |
| LLM (프로필 생성) | Ollama `qwen2.5:7b` |
| MCP 프레임워크 | FastMCP |
| 언어 | Python 3.x |

---

## 설치 및 초기 설정

### 1. 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Ollama 설치 및 모델 다운로드

```bash
# https://ollama.com 에서 Ollama 설치 후
ollama pull qwen2.5:7b
```

### 3. ChromaDB 초기 빌드

대화 파일이 `conversations/` 폴더에 있을 때 1회 실행합니다.

```bash
python memory_pipeline.py ./conversations ./memory_db
```

### 4. 사용자 프로필 생성

```bash
python generate_profile.py
```

---

## Claude Desktop 연동

`claude_desktop_config.json`에 아래 설정을 추가합니다:

```json
{
  "mcpServers": {
    "memory-layer": {
      "command": "C:/path/to/vLLM_memory_layer_ver2/.venv/Scripts/python.exe",
      "args": ["C:/path/to/vLLM_memory_layer_ver2/mcp_server.py"]
    }
  }
}
```

> **경로는 실제 설치 위치에 맞게 수정하세요.**

---

## 대화 파일 형식

`conversations/` 폴더에 아래 형식의 마크다운 파일을 저장합니다:

```
파일명: YYYY-MM-DD_주제.md
예시: 2025-06-24_SaaS화 대체 전략.md
```

파일 내 대화 구조:

```markdown
- 생성일: 2025-06-24
- ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

## 👤 User

사용자 질문 내용...

## 🤖 Assistant

Claude 답변 내용...
```

---

## 사용 흐름

```
세션 시작   → get_profile()           — 사용자 맥락 파악
대화 중     → search_memory_tool()    — 관련 과거 대화 참조
날짜 질문   → get_conversation_by_date() — 특정 날짜 대화 조회
세션 종료   → add_memory(file_path)   — 새 대화 DB에 저장
주기적으로  → update_profiles()       — 프로필 자동 갱신
```

---

## 파일 구조

```
vLLM_memory_layer_ver2/
├── mcp_server.py          # FastMCP 서버 (메인 진입점)
├── memory_pipeline.py     # MD 파싱 + 임베딩 + ChromaDB 파이프라인
├── generate_profile.py    # 초기 사용자 프로필 생성
├── test_memory.py         # 메모리 검색 테스트
├── test_embeddings.py     # 임베딩 모델 비교 테스트
├── requirements.txt       # Python 의존성
├── conversations/         # 대화 마크다운 파일 저장소
├── memory/                # 사용자 프로필 (5개 MD 파일)
└── memory_db/             # ChromaDB 벡터 DB 영구 저장소
```

---

## 참고 사항

- 임베딩 모델은 서버 시작 시 1회만 로드됩니다 (매 호출마다 로드하지 않음)
- 회사 내부망 self-signed 인증서 환경을 위한 SSL 우회 처리가 포함되어 있습니다
- 모든 처리가 로컬에서 이루어지므로 대화 내용이 외부로 전송되지 않습니다
- 대화 파일 인코딩은 UTF-8, CP949, EUC-KR을 자동 감지합니다
