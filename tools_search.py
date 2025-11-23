# tools_search.py
# LangChain이 사용하는 도구 중 하나로,
# 전체 사이트에서 검색어로 결과를 찾게 해준다.
import requests
from bs4 import BeautifulSoup

from langchain_core.tools import tool
from indexer import search

@tool
def search_site(query: str) -> str:
    """과학관 전체 사이트에서 키워드 검색"""
    hits = search(query, limit=8)
    if not hits:
        return "Observation:\n\n검색 결과 없음."
    # 검색 결과를 한 줄씩 정리
    lines = [f"- {h['title']} | {h['url']}\n  {h['snippet']}" for h in hits]
    return "Observation:\n\n" + "\n".join(lines)
