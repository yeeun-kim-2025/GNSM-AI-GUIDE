# tools_notice.py
# ------------------------------------------------------------
# LangChain/LangGraph에서 호출하는 "공지 전용" 도구 2개.
# - search_notices_tool: 키워드로 공지를 찾기
# - latest_notices_tool: 최근 공지 N건 간단 목록
# 결과는 "Observation:\n\n" 접두사로 반환(신승헌 패턴 호환).
# ------------------------------------------------------------

from langchain_core.tools import tool
from notice_indexer import search_notices, latest_notices

@tool
def search_notices_tool(query: str) -> str:
    """한국어 키워드로 과학관 '공지사항' 전용 검색."""
    hits = search_notices(query, limit=8)
    if not hits:
        return "Observation:\n\n검색 결과 없음."
    lines = [
        f"- {h['title']} ({h['date']}) | {h['url']}\n  {h['snippet']}"
        for h in hits
    ]
    return "Observation:\n\n" + "\n".join(lines)

@tool
def latest_notices_tool(n: int = 8) -> str:
    """최근 색인된 공지 N건(제목/날짜/링크) 간단 목록."""
    rows = latest_notices(limit=int(n))
    if not rows:
        return "Observation:\n\n최근 공지 없음."
    lines = [f"- {r['title']} ({r['date']}) | {r['url']}" for r in rows]
    return "Observation:\n\n" + "\n".join(lines)
