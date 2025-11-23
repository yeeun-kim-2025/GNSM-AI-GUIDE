# utils.py — Grounded LLM + 오타 자동 보정 + 주제 버튼 버전
# 실시간 requests + HTML → FACTS + 사이트검색 + LLM(FACTS만 요약/말투)

import os
import traceback
import logging
import re
import difflib

import requests
import streamlit as st
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

MUSEUM_BASE_URL = "https://www.sciencecenter.go.kr"


# ---------------------------------------
# 0. 로거
# ---------------------------------------
def init_logger():
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


init_logger()


# ---------------------------------------
# 1. 공통 유틸: 시간 표현 보정, 제목 추출
# ---------------------------------------
_TIME_RANGE_PATTERN = re.compile(r"(\d{1,2}:\d{2})\s*(~|∼|-|–|—)?\s*(\d{1,2}:\d{2})")


def _normalize_time_ranges(text: str) -> str:
    """
    HTML에서 시간 구간이 '10:0010:40' 처럼 붙어 있는 경우
    '10:00~10:40' 형태로 자동 보정한다.
    이미 ~, ∼, -, –, — 등이 들어있는 경우는 그대로 둔다.
    """

    def repl(m: re.Match):
        start, sep, end = m.group(1), m.group(2), m.group(3)
        # 이미 구분자가 있으면 그대로 사용
        if sep and sep.strip():
            return f"{start}{sep}{end}"
        # 구분자가 없으면 ~로 채워준다
        return f"{start}~{end}"

    return _TIME_RANGE_PATTERN.sub(repl, text)


def _extract_page_title(html: str) -> str:
    """
    페이지의 제목 후보를 추출한다.
    - 공지 본문 상단 h1/h2/h3
    - 없으면 <title> 태그
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in ["h1", "h2", "h3"]:
        el = soup.find(tag)
        if el and el.get_text(strip=True):
            return " ".join(el.stripped_strings)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


# ---------------------------------------
# 1-1. LLM 답변 마크다운 정리 (취소선 제거)
# ---------------------------------------
def _cleanup_answer_markdown(answer: str) -> str:
    """
    LLM 답변 안에 들어간 ~~취소선~~ 마크다운을 제거한다.
    - 20:00~~21:30 → 20:00~21:30
    - ~~문장~~ → 문장
    """
    # 시간 구간에 잘못 들어간 ~~ 를 한 개의 ~ 로 통일
    answer = re.sub(
        r"(\d{1,2}:\d{2})\s*~~\s*(\d{1,2}:\d{2})",
        r"\1~\2",
        answer,
    )
    # 일반적인 ~~텍스트~~ 취소선 제거
    answer = re.sub(r"~~([^~]+?)~~", r"\1", answer)
    return answer


# ---------------------------------------
# 2. HTML → FACTS (텍스트/표/이미지)
# ---------------------------------------
def _find_html_with_table(obj):
    """JSON / dict / list 안에서 <table>이 들어있는 HTML 문자열을 재귀적으로 찾기."""
    if isinstance(obj, str) and "<table" in obj.lower():
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_html_with_table(v)
            if found:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _find_html_with_table(v)
            if found:
                return found
    return None


def _extract_tables_from_html(html: str, max_tables: int = 10) -> str:
    """HTML 안의 <table>들을 마크다운 표로 변환 (모든 행/열 포함)."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return ""

    blocks = []
    for table in tables[:max_tables]:
        rows = []
        for tr in table.find_all("tr"):
            row = []
            for td in tr.find_all(["th", "td"]):
                links = td.find_all("a")
                if links:
                    parts = []
                    for a in links:
                        text = " ".join(a.stripped_strings) or "자세히 보기"
                        href = (a.get("href") or "").strip()
                        if href.startswith("/"):
                            href = MUSEUM_BASE_URL + href
                        if href:
                            parts.append(f"[{text}]({href})")
                        else:
                            parts.append(text)
                    cell_text = " ".join(parts)
                else:
                    cell_text = " ".join(td.stripped_strings)

                # 시간 구간 표기 보정 (10:0010:40 → 10:00~10:40)
                cell_text = _normalize_time_ranges(cell_text)

                row.append(cell_text)
            if row:
                rows.append(row)

        if not rows:
            continue

        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]

        header = rows[0]
        body = rows[1:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        for r in body:
            lines.append("| " + " | ".join(r) + " |")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _extract_tables_and_images_for_display(html: str):
    """
    HTML 전체에서 표와 이미지 URL을 별도로 추출한다.
    (지금은 '존재 여부'만 확인해서 안내 문구를 띄우는 데 사용)
    """
    soup = BeautifulSoup(html, "lxml")
    # 스크립트/스타일 등 제거
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()

    tables_md = _extract_tables_from_html(str(soup))

    image_urls = []
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if not src:
            continue
        if src.startswith("/"):
            src = MUSEUM_BASE_URL + src
        if src not in image_urls:
            image_urls.append(src)

    return tables_md, image_urls


def _html_to_facts(html: str) -> str:
    """
    HTML 전체를 FACTS로 변환.
    - <h1~h4>, <p>, <li>를 한 줄씩 정리해서 구조를 최대한 살림
    - 표는 마크다운 표로 전체 추출
    - 이미지 src도 모두 FACTS에 포함
    - 중간에서 자르지 않고 전체 사용
    """
    soup = BeautifulSoup(html, "lxml")

    # 스크립트/스타일 등 제거
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()

    # 텍스트: 블록 요소별로 한 줄씩
    lines = []
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(elem.stripped_strings)
        if not text:
            continue

        text = _normalize_time_ranges(text)

        if elem.name == "li":
            lines.append(f"- {text}")
        elif elem.name in ["h1", "h2", "h3", "h4"]:
            level = int(elem.name[1])
            level = min(level, 4)
            prefix = "#" * level
            lines.append(f"{prefix} {text}")
        else:
            lines.append(text)

    text_block = "\n".join(lines)

    # 표
    tables_md = _extract_tables_from_html(str(soup))

    # 이미지 (FACTS용 URL 나열)
    image_urls = []
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if not src:
            continue
        if src.startswith("/"):
            src = MUSEUM_BASE_URL + src
        if src not in image_urls:
            image_urls.append(src)

    parts = []
    if text_block:
        parts.append("### 텍스트\n" + text_block)
    if tables_md:
        parts.append("### 표\n" + tables_md)
    if image_urls:
        parts.append("### 이미지 URL\n" + "\n".join(image_urls))

    if not parts:
        return "이 페이지에서 텍스트나 표를 찾지 못했습니다."

    return "\n\n".join(parts)


# ---------------------------------------
# 3. 실시간 페이지 fetch (showBoard API + 일반 GET)
# ---------------------------------------
def _fetch_page(url: str) -> dict:
    """
    url에서 내용을 가져와서
    {
      "source": url,
      "title": 페이지 제목(있으면),
      "facts": FACTS 텍스트,
      "has_rich": 표나 이미지가 하나라도 있으면 True,
      혹은 "error": "..."
    } 형태로 반환.
    """
    logging.info(f"[LIVE] URL 가져오기: {url}")
    try:
        # 공지 상세: 내부 showBoard JSON 먼저 시도
        m = re.search(r"/scipia/introduce/notice/(\d+)", url)
        if m:
            board_id = m.group(1)
            api_url = f"{MUSEUM_BASE_URL}/scipia/boards/showBoard/{board_id}"
            headers = {
                "User-Agent": "GNSM-AI-Docent/1.0",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }
            resp = requests.post(api_url, headers=headers, json={}, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            html = _find_html_with_table(data)
            if not html and isinstance(data, dict) and isinstance(data.get("content"), str):
                html = data["content"]

            title = ""
            if isinstance(data, dict):
                for k in ["title", "subject", "boardTitle"]:
                    if k in data and isinstance(data[k], str):
                        title = data[k].strip()
                        break

            if html:
                facts = _html_to_facts(html)
                tables_md, image_urls = _extract_tables_and_images_for_display(html)
                if not title:
                    title = _extract_page_title(html)
                has_rich = bool(tables_md or image_urls)
                return {
                    "source": url,
                    "title": title,
                    "facts": facts,
                    "has_rich": has_rich,
                }

        # 그 외는 일반 GET
        resp = requests.get(
            url,
            headers={"User-Agent": "GNSM-AI-Docent/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        html = resp.text
        facts = _html_to_facts(html)
        tables_md, image_urls = _extract_tables_and_images_for_display(html)
        title = _extract_page_title(html)
        has_rich = bool(tables_md or image_urls)

        return {
            "source": url,
            "title": title,
            "facts": facts,
            "has_rich": has_rich,
        }

    except Exception:
        logging.error(f"[LIVE] 페이지 로드 실패:\n{traceback.format_exc()}")
        return {"source": url, "error": "페이지 로드 실패"}


# ---------------------------------------
# 4. 과학관 주요 페이지 매핑
# ---------------------------------------
LIVE_PAGES = {
    # --- 핵심 운영/예약 ---
    "홈페이지": f"{MUSEUM_BASE_URL}/",
    "공지사항": f"{MUSEUM_BASE_URL}/scipia/introduce/notice",
    "이용안내": f"{MUSEUM_BASE_URL}/scipia/guide/totalGuide",
    "주차안내": f"{MUSEUM_BASE_URL}/scipia/introduce/parking",
    "연간회원": f"{MUSEUM_BASE_URL}/scipia/guide/paidMember",
    "단체관람": f"{MUSEUM_BASE_URL}/scipia/guide/groupTours",
    "추천관람코스": f"{MUSEUM_BASE_URL}/scipia/guide/recommendCourse",
    "관람객대피": f"{MUSEUM_BASE_URL}/scipia/communication/safety",
    "안전사고수칙": f"{MUSEUM_BASE_URL}/scipia/communication/safetyRule",
    "편의시설": f"{MUSEUM_BASE_URL}/scipia/guide/convenience",
    "식음시설": f"{MUSEUM_BASE_URL}/scipia/guide/food",
    "교통안내": f"{MUSEUM_BASE_URL}/scipia/introduce/location",

    # --- 행사 및 공연 ---
    "행사": f"{MUSEUM_BASE_URL}/scipia/events/list/culture",
    "공연": f"{MUSEUM_BASE_URL}/scipia/events/list/play",

    # --- 천문우주관 ---
    "천체투영관 소개": f"{MUSEUM_BASE_URL}/scipia/display/planetarium",
    "천체투영관 운영": f"{MUSEUM_BASE_URL}/scipia/introduce/notice/24281",
    "천체투영관 프로그램": f"{MUSEUM_BASE_URL}/scipia/introduce/notice/24281",
    "천체투영관 예약": f"{MUSEUM_BASE_URL}/scipia/schedules?ACADEMY_CD=ACD007&CLASS_CD=CL7001",
    "천체투영관 단체": f"{MUSEUM_BASE_URL}/scipia/introduce/notice/23441",

    "천문대 소개": f"{MUSEUM_BASE_URL}/scipia/display/planetarium/observation",
    "천문대 운영": f"{MUSEUM_BASE_URL}/scipia/introduce/notice/25098",
    "천문대 프로그램": f"{MUSEUM_BASE_URL}/scipia/introduce/notice/25098",
    "천문대 예약": f"{MUSEUM_BASE_URL}/scipia/schedules?ACADEMY_CD=ACD007&CLASS_CD=CL7003",
    "천문대 단체": f"{MUSEUM_BASE_URL}/scipia/introduce/notice/25100",

    "스페이스 아날로그 프로그램": f"{MUSEUM_BASE_URL}/scipia/display/planetarium/spaceAnalog",
    "스페이스 아날로그 예약": f"{MUSEUM_BASE_URL}/scipia/schedules?ACADEMY_CD=ACD007&CLASS_CD=CL7002",
    "스페이스 아날로그 단체": f"{MUSEUM_BASE_URL}/scipia/introduce/notice/24400",

    # --- 상설전시관 ---
    "자연사관": f"{MUSEUM_BASE_URL}/scipia/display/mainBuilding/naturalHistory",
    "첨단기술관": f"{MUSEUM_BASE_URL}/scipia/display/mainBuilding/advancedTechnology2",
    "과학탐구관": f"{MUSEUM_BASE_URL}/scipia/display/mainBuilding/basicScience",
    "한국문명관": f"{MUSEUM_BASE_URL}/scipia/display/mainBuilding/traditionalSciences",
    "미래상상SF관": f"{MUSEUM_BASE_URL}/scipia/display/mainBuilding/sfSpecial",
    "유아체험관": f"{MUSEUM_BASE_URL}/scipia/display/mainBuilding/kidsPlayground",
    "명예의전당": f"{MUSEUM_BASE_URL}/scipia/display/frontier/hallOfFame",
    "특별기획전": f"{MUSEUM_BASE_URL}/scipia/events/list/exhibition#n",

    # --- 전시연계 ---
    "체험전시물 예약": f"{MUSEUM_BASE_URL}/scipia/display/displayExperience",
    "전시장 프로그램 안내": f"{MUSEUM_BASE_URL}/scipia/introduce/notice/25399",
    "전시해설": f"{MUSEUM_BASE_URL}/scipia/display/displayExplanation",

    # --- 야외전시관 ---
    "곤충생태관": f"{MUSEUM_BASE_URL}/scipia/display/outdoorEcological/insectarium",
    "생태공원": f"{MUSEUM_BASE_URL}/scipia/display/outdoorEcological/ecoPark",
    "공룡공원": f"{MUSEUM_BASE_URL}/scipia/display/outdoorEcological/dinosaurAndHistory",
    "옥외전시장": f"{MUSEUM_BASE_URL}/scipia/display/outdoorEcological/outdoor",

    # --- 소통 및 소개 ---
    "인사말": f"{MUSEUM_BASE_URL}/scipia/introduce/chief",
    "연혁": f"{MUSEUM_BASE_URL}/scipia/introduce/history",
    "조직 및 연혁": f"{MUSEUM_BASE_URL}/scipia/introduce/organization",
    "주변시설": f"{MUSEUM_BASE_URL}/scipia/introduce/surround",
    "유관기관": f"{MUSEUM_BASE_URL}/scipia/introduce/familySites",
    "수도권과학관": f"{MUSEUM_BASE_URL}/scipia/introduce/capitalScience",
    "보도자료": f"{MUSEUM_BASE_URL}/scipia/introduce/report",
    "현장스케치": f"{MUSEUM_BASE_URL}/scipia/introduce/sketch",
    "채용공고": f"{MUSEUM_BASE_URL}/scipia/introduce/recruit",
    "일반자료실": f"{MUSEUM_BASE_URL}/scipia/communication/normalLibrary",
    "규정자료실": f"{MUSEUM_BASE_URL}/scipia/communication/roleLibrary",
    "자주묻는질문": f"{MUSEUM_BASE_URL}/scipia/communication/faq/faqTotal",
    "의견수렴": f"{MUSEUM_BASE_URL}/scipia/communication/opinions",
    "과학자료실": f"{MUSEUM_BASE_URL}/scipia/references",
    "자원봉사": f"{MUSEUM_BASE_URL}/scipia/schedules/voluntary",
}


# ---------------------------------------
# 5. 주제 트리 (대분류 -> 중/소분류)
# ---------------------------------------
TOPIC_TREE = {
    "guide": {
        "label": "관람 이용안내",
        "children": [
            ("이용안내 전체", "이용안내(관람시간, 요금, 휴관일) 알려줘"),
            ("관람요금", "관람요금 알려줘"),
            ("휴관일/운영일", "휴관일/운영일 알려줘"),
            ("연간회원", "연간회원 안내해줘"),
            ("주차안내", "주차안내 알려줘"),
            ("교통안내", "교통안내 알려줘"),
        ],
    },
    "astro_program": {
        "label": "천문우주시설",
        "children": [
            ("천체투영관 프로그램", "천체투영관 프로그램 알려줘"),
            ("천체투영관 단체", "천체투영관 단체 프로그램 안내해줘"),
            ("천문대 프로그램", "천문대 프로그램 알려줘"),
            ("천문대 단체", "천문대 단체 프로그램 안내해줘"),
            ("스페이스 아날로그 프로그램", "스페이스 아날로그 프로그램 알려줘"),
            ("스페이스 아날로그 단체", "스페이스 아날로그 프로그램 알려줘"),
        ],
    },
    "exhibition": {
        "label": "상설전시관",
        "children": [
            ("자연사관", "자연사관 안내해줘"),
            ("첨단기술관", "첨단기술관 안내해줘"),
            ("과학탐구관", "과학탐구관 안내해줘"),
            ("한국문명관", "한국문명관 안내해줘"),
            ("미래상상SF관", "미래상상SF관 안내해줘"),
            ("유아체험관", "유아체험관 안내해줘"),
            ("명예의전당", "명예의 전당 안내해줘"),
            ("특별기획전", "특별기획전 안내해줘"),
        ],
    },
    "group_program": {
        "label": "단체·전시연계 프로그램",
        "children": [
            ("단체관람", "단체관람 안내해줘"),
            ("전시해설", "전시해설 프로그램 안내해줘"),
            ("체험전시물 예약", "체험전시물 예약 안내해줘"),
        ],
    },
    "outdoor": {
        "label": "야외전시·생태",
        "children": [
            ("곤충생태관", "곤충생태관 안내해줘"),
            ("공룡공원", "공룡공원 안내해줘"),
            ("생태공원", "생태공원 안내해줘"),
            ("옥외전시장", "옥외전시장 안내해줘"),
        ],
    },
    "facility": {
        "label": "편의시설·식음·교통",
        "children": [
            ("식음시설", "식음시설(카페·식당) 안내해줘"),
            ("편의시설", "편의시설 안내해줘"),
            ("주차안내", "주차안내 알려줘"),
            ("교통안내", "교통안내 알려줘"),
        ],
    },
    "etc": {
        "label": "FAQ·공지·기타",
        "children": [
            ("자주묻는질문", "자주 묻는 질문 알려줘"),
            ("공지사항", "공지사항 안내해줘"),
            ("과학자료실", "과학자료실 안내해줘"),
        ],
    },
}


# ---------------------------------------
# 6. 오타/유사도 기반 LIVE_PAGES 키워드 추정
# ---------------------------------------
def _guess_live_key(question: str, cutoff: float = 0.6):
    q = re.sub(r"\s+", "", question)  # 공백 제거

    best_key = None
    best_score = 0.0

    for key in LIVE_PAGES.keys():
        key_norm = key.replace(" ", "")

        score = difflib.SequenceMatcher(None, key_norm, q).ratio()

        if key_norm in q or q in key_norm:
            score = max(score, 0.99)

        if score > best_score:
            best_score = score
            best_key = key

    if best_key and best_score >= cutoff:
        return best_key

    return None


def _match_live_keys(question: str):
    q = re.sub(r"\s+", "", question)
    matched = []

    for key in LIVE_PAGES.keys():
        if key.replace(" ", "") in q:
            matched.append(key)

    if not matched:
        guessed = _guess_live_key(question)
        if guessed:
            matched.append(guessed)
            st.info(f"혹시 **'{guessed}'**(을)를 의미하신 건가요? 해당 페이지 기준으로 안내드릴게요.")

    time_fee_keywords = [
        "운영시간",
        "관람시간",
        "개관시간",
        "폐관시간",
        "관람요금",
        "관람료",
        "입장료",
        "요금",
        "관람일",
        "운영일",
        "개관일",
        "휴관일",
        "휴무일",
        "휴관",
        "휴무",
        "운영일정",
        "개관일정",
    ]
    if any(kw in q for kw in time_fee_keywords):
        if "이용안내" not in matched:
            matched.append("이용안내")

    return matched


# ---------------------------------------
# 7. 사이트 전체 검색 (indexer 제거 버전: 현재는 사용 안 함)
# ---------------------------------------
def _search_site(query: str, limit: int = 5):
    """
    예전에는 indexer 기반 전체 검색을 사용했지만,
    현재 배포 버전에서는 indexer를 제거했으므로 빈 결과를 반환합니다.
    (필요하면 나중에 다른 검색 방식으로 교체)
    """
    logging.info(f"[SEARCH] (indexer 제거) 검색 호출: {query} (limit={limit})")
    return []


# ---------------------------------------
# 8. LLM 시스템 프롬프트 & 푸터
# ---------------------------------------
STRICT_SYSTEM_PROMPT = """
당신은 국립과천과학관 전용 AI 가이드입니다.

[역할]
- 사용자가 묻는 내용을, 아래 FACTS에 포함된 정보만 사용해서 이해하기 쉽게 설명합니다.
- FACTS는 국립과천과학관 공식 홈페이지에서 가져온 실제 내용입니다.

[엄격한 규칙]
1. FACTS 블록 안의 문장을 그대로 길게 복사·붙여넣기 하지 마세요.
2. FACTS 블록(예: '### 텍스트', '### 표', '[섹션:' 등)의 구조나 문구를
   답변에 그대로 보여주지 마세요.
3. FACTS에서 필요한 정보만 뽑아서 **짧은 불릿/표 형태로 정리만** 해주세요.
4. 답변은 최대 15줄 이내로, 각 줄은 한두 문장으로만 작성하세요.
5. FACTS에 없는 정보(숫자, 날짜, 요금, 시간, 프로그램명 등)는 절대로 추가하지 마세요.
6. FACTS에 없는 부분이 있더라도 해당 부분은 생략하고,
   FACTS에서 확인 가능한 정보만 정리해서 보여주세요.
7. 사용자에게 "FACTS 없음", "홈페이지에서 찾을 수 없습니다" 같은
   오류 안내 문구를 출력하지 마세요. (그 문구는 코드에서만 사용합니다.)

[출력 형식]
- 항상 마크다운(Markdown) 형식으로 작성하세요.
- 첫 줄에는 간단한 제목을 `### 제목` 형식으로 작성하세요.
- 그 아래는 줄글이 아니라 핵심 항목을 불릿(`- 항목`)이나 간단한 표로 정리하세요.
- 운영시간/요금/대상/참가인원/예약방법/프로그램 내용을 각각 항목별로 분리해서 써 주세요.
"""


def _append_info_footer(answer: str) -> str:
    """
    이전 버전의 '안내드립니다!' 공통 문구는 제거.
    지금은 아무 것도 추가하지 않고 그대로 반환.
    """
    return answer


# ---------------------------------------
# 9. LLM 초기화
# ---------------------------------------
def _init_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI API Key가 없습니다.")
    return ChatOpenAI(
        temperature=temperature,
        model=model_name,
        openai_api_key=api_key,
    )


# ---------------------------------------
# 10. 주제 버튼 + 돌아가기
# ---------------------------------------
def _render_topic_shortcuts():
    stage = st.session_state.get("topic_stage", "root")
    group = st.session_state.get("topic_group", None)

    st.markdown("#### 🔍 무엇을 도와드릴까요?")

    if stage == "root" or group not in TOPIC_TREE:
        cols = st.columns(3)
        for i, (group_key, meta) in enumerate(TOPIC_TREE.items()):
            label = meta["label"]
            col = cols[i % 3]
            with col:
                if st.button(label, key=f"topic_root_{group_key}"):
                    st.session_state["topic_stage"] = "mid"
                    st.session_state["topic_group"] = group_key
                    st.rerun()
        return

    meta = TOPIC_TREE[group]
    st.markdown(f"##### 📌 '{meta['label']}'에서 더 궁금한 내용을 골라보세요")

    children = meta.get("children", [])
    cols = st.columns(3)
    for i, (label, query) in enumerate(children):
        col = cols[i % 3]
        with col:
            if st.button(label, key=f"topic_child_{group}_{i}"):
                st.session_state["pending_query"] = query

    back_col, _ = st.columns([1, 3])
    with back_col:
        if st.button("⬅ 돌아가기", key="topic_back_root", type="primary"):
            st.session_state["topic_stage"] = "root"
            st.session_state["topic_group"] = None
            st.session_state["pending_query"] = ""
            st.rerun()


def _render_global_back_button():
    if st.button("⬅ 돌아가기", key="global_back_to_topics", type="primary"):
        st.session_state["topic_stage"] = "root"
        st.session_state["topic_group"] = None
        st.session_state["pending_query"] = ""
        st.rerun()


# ---------------------------------------
# 11. Streamlit 메인 함수
# ---------------------------------------
def run_chat_assistant(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    system_prompt: str = STRICT_SYSTEM_PROMPT,
) -> None:
    """
    Grounded LLM 구조:
      1) 홈페이지에서 FACTS(전체 내용)를 모은다.
      2) FACTS만 LLM에 넘겨서 말투/요약만 수행 (추가 정보 금지)
      3) 출처 페이지에 표/이미지가 있으면
         → 답변 뒤, '더 자세히 보기' 버튼 앞에
           '홈페이지 안내사항을 꼭 함께 보라'는 안내 문구를 보여준다.
    """

    # 세션 상태 초기화
    if "topic_stage" not in st.session_state:
        st.session_state["topic_stage"] = "root"
    if "topic_group" not in st.session_state:
        st.session_state["topic_group"] = None
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = ""
    if "messages" not in st.session_state:
        # 👉 인사말은 messages에 넣지 않고, 화면에만 한 번 그린다.
        st.session_state.messages = []

    # LLM 초기화
    try:
        llm = _init_llm(model_name=model_name, temperature=temperature)
    except Exception as e:  # pragma: no cover
        st.error(f"⚠️ LLM 초기화 실패: {e}")
        return

    # 인사말: 대화 내역이 하나도 없을 때만 한 번 보여주기
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("#### 안녕하세요! 국립과천과학관 AI 가이드입니다 🤖\n\n")

    # 기존 대화 출력
    for msg in st.session_state.messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            if role == "assistant":
                st.markdown(msg.content, unsafe_allow_html=True)
            else:
                st.markdown(msg.content)

    st.markdown("---")
    _render_topic_shortcuts()

    pending = st.session_state.get("pending_query", "")
    user_input = st.chat_input("궁금한 내용을 입력해 주세요 (예: 천문대 단체 프로그램 알려줘)")

    if pending:
        user_msg = pending
        st.session_state["pending_query"] = ""
    elif user_input:
        user_msg = user_input
    else:
        return

    user_msg_obj = HumanMessage(content=user_msg)
    st.session_state.messages.append(user_msg_obj)
    with st.chat_message("user"):
        st.markdown(user_msg)

    # 1단계: FACTS + "표/이미지 존재 여부" 수집
    facts_sections = []
    link_items = []
    has_rich_content = False  # 어떤 출처든 표나 이미지가 있으면 True

    # (1) LIVE_PAGES
    matched_keys = _match_live_keys(user_msg)
    for key in matched_keys:
        url = LIVE_PAGES[key]
        page_data = _fetch_page(url)
        src = page_data.get("source", url)
        title = page_data.get("title") or key

        link_items.append((title, src))

        if page_data.get("has_rich"):
            has_rich_content = True

        if "facts" in page_data:
            section = f"[섹션: {title}]\n{page_data['facts']}"
        else:
            section = (
                f"[섹션: {title}]\n"
                "이 페이지는 자동으로 내용을 불러오지 못했습니다. "
                f"아래 링크를 눌러 홈페이지에서 직접 확인해 주세요. ({src})"
            )
        facts_sections.append(section)

    # (2) 사이트 전체 검색 (현재 indexer 제거로 인해 항상 빈 결과)
    hits = _search_site(user_msg, limit=3)
    for h in hits:
        url = h.get("url") or ""
        raw_title = h.get("title") or "관련 페이지"
        snippet = h.get("snippet") or ""

        page_data = None
        if url:
            page_data = _fetch_page(url)
            src = page_data.get("source", url)
        else:
            src = url

        display_title = raw_title
        if page_data is not None:
            display_title = page_data.get("title") or raw_title

        if url and all(u != url for _, u in link_items):
            link_items.append((display_title, url))

        if page_data is not None and page_data.get("has_rich"):
            has_rich_content = True

        if url and page_data is not None and "facts" in page_data:
            section = f"[섹션: {display_title}]\n{page_data['facts']}"
        else:
            body = snippet or "홈페이지에 관련 페이지가 있습니다. 링크를 눌러 내용을 확인해 주세요."
            section = f"[섹션: {display_title}]\n{body}"
        facts_sections.append(section)

    facts_text = "\n\n----------------\n\n".join(facts_sections).strip()

    if not facts_text:
        body_lines = [
            "### 홈페이지에서 직접 확인이 필요한 내용입니다.",
            "",
            "- 질문과 정확히 연결되는 정보를 자동으로 찾기 어렵습니다.",
            "- 질문을 조금 더 구체적으로, 또는 다른 표현으로 다시 입력해 주세요 😊",
            "",
            f"- [이용안내 메인]({LIVE_PAGES['이용안내']})",
            f"- [전체 메뉴 한눈에 보기]({MUSEUM_BASE_URL}/scipia/introduce/siteMap)",
        ]
        body = "\n".join(body_lines)
        answer = _append_info_footer(body)
        with st.chat_message("assistant"):
            st.markdown(answer, unsafe_allow_html=True)
            _render_global_back_button()
        st.session_state.messages.append(AIMessage(content=answer))
        return

    # 2단계: LLM 호출
    user_prompt = (
        "사용자 질문:\n"
        f"{user_msg}\n\n"
        "아래는 국립과천과학관 홈페이지에서 가져온 FACTS(원문 데이터)입니다.\n"
        "- 이 FACTS는 내부 참고용이며, 사용자에게 그대로 보여주면 안 됩니다.\n"
        "- FACTS 블록 속 문장과 제목(예: '### 텍스트', '### 표', '[섹션:' 등)을 복사하지 말고,\n"
        "  필요한 정보만 뽑아서 불릿/표로 짧게 정리해서 보여 주세요.\n\n"
        "FACTS 시작\n"
        "----------------\n"
        f"{facts_text}\n"
        "----------------\n"
        "FACTS 끝\n"
    )

    messages_for_llm = [
        SystemMessage(content=STRICT_SYSTEM_PROMPT),
        SystemMessage(
            content=(
                "추가 지시(STRICT 규칙과 모순되지 않는 범위에서만 따르세요):\n"
                f"{system_prompt}"
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    with st.chat_message("assistant"):
        with st.spinner("잠시만 기다려주세요!"):
            try:
                answer_obj = llm.invoke(messages_for_llm)
                answer_text = answer_obj.content
            except Exception as e:  # pragma: no cover
                answer_text = f"⚠️ 응답 생성 중 오류 발생: {e}"

            # 👉 취소선/시간표기 정리
            answer_text = _cleanup_answer_markdown(answer_text)

            # 3단계: "홈페이지 꼭 보기" 안내 문구 (표/이미지 있을 때만)
            rich_notice_md = ""
            if has_rich_content:
                rich_notice_md = (
                    "\n\n> ℹ️ **더욱 자세한 안내를 원하신다면?**  \n"
                    "> 홈페이지를 함께 확인해주세요!\n"
                )

            # 4단계: '홈페이지 확인하기' 버튼
            links_md = ""
            if link_items:
                first_label, first_url = link_items[0]
                links_md = f"""
<a href="{first_url}" target="_blank" style="text-decoration:none;">
    <button style="padding:8px 16px; font-size:16px; border-radius:6px; border:1px solid #00519A; background-color:#00519A; color:white; cursor:pointer;">
        🔎 홈페이지 확인하기
    </button>
</a>
"""

            final_answer = answer_text + rich_notice_md + "\n\n" + links_md
            final_answer = _append_info_footer(final_answer)

            st.markdown(final_answer, unsafe_allow_html=True)
            _render_global_back_button()
            st.session_state.messages.append(AIMessage(content=final_answer))
