"""任务第 1 部分：人物公开资料的“混合式”自动获取流水线。

    fetch    下载原始资料：Wikidata（SPARQL 结构化查询）、Wikipedia 源文本、SEC 文件、新闻报道 → 快照
    extract  从快照中抽取候选事实（三种方法，全部保存证据）：
               ① Wikidata 结构化字段 + 句子模板（生平：出生、学历、任职、创办机构、获奖）
               ② 规则（正则）抽取 SEC 年报中的财务数字
               ③ 本地 Qwen 从文章中抽取事实，必须附“逐字引用”，程序再核对引用是否原样出现在原文中
               另有少量人工补充条目（personas/manual_facts.csv），同样要逐字核对引用
    build    按公开的筛选规则（时间截止、结果黑名单、类别白名单、去重、数量上限）选出事实，
             生成 personas/candidate_facts.csv（全部候选 + 入选/淘汰原因）、personas/sources.csv（入选事实）、
             personas/bezos.txt 与 personas/graham.txt（写进 prompt 的人物资料）
    all      依次执行以上三步

用法（在仓库根目录）：
    python experiments/bezos_graham/persona_pipeline.py fetch --email 你的邮箱
    python experiments/bezos_graham/persona_pipeline.py extract --backend gguf --model-path D:\\models\\Qwen3-4B-Instruct-2507-Q8_0.gguf
    python experiments/bezos_graham/persona_pipeline.py build
    # 不联网：跳过 fetch，extract/build 只用已保存的快照与已缓存的模型输出
    # 不跑模型：extract --llm none（只做 ① ② 和人工条目）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

EXP_DIR = Path(__file__).resolve().parent
PERSONA_DIR = EXP_DIR / "personas"
EVIDENCE_DIR = PERSONA_DIR / "evidence"
SNAPSHOT_DIR = EVIDENCE_DIR / "snapshots"
LLM_DIR = EVIDENCE_DIR / "llm_extractions"
CANDIDATES_CSV = PERSONA_DIR / "candidate_facts.csv"
SOURCES_CSV = PERSONA_DIR / "sources.csv"
MANUAL_CSV = PERSONA_DIR / "manual_facts.csv"

# ===========================================================================
# 公开的筛选规则（报告里可以直接引用）
# ===========================================================================
CUTOFF = "2013-07-31"          # 模拟的谈判发生在 2013 年 7 月：只用这之前就成立的事实
CUTOFF_YEAR = 2013
# 结果黑名单：任何透露真实交易结果/价格的内容都不得进入 prompt
OUTCOME_BLACKLIST = re.compile(
    r"250 million|purchase price|\bsold\b|\bsale\b|\bto sell\b|\bselling\b|Nash Holdings|"
    r"agreed to (buy|acquire|sell)|\bacquisition of the (post|paper|newspaper)\b|new owner|"
    r"Graham Holdings",            # 公司在出售报纸后（2013 年 11 月）才改名为 Graham Holdings
    re.IGNORECASE)
LLM_CATEGORIES = {"background", "education", "career", "philosophy", "views_on_post"}
MAX_FACTS_PER_PERSON = 12
DEDUP_JACCARD = 0.6

PEOPLE = {
    "Jeff Bezos": {"qid": "Q312556", "file": "bezos.txt"},
    "Donald Graham": {"qid": "Q5294282", "file": "graham.txt"},
}

# 文本来源。Wikipedia 用 action=raw 取源文本（本地清理成纯文本）；其余直接下载 HTML。
TEXT_SOURCES: Dict[str, Dict[str, str]] = {
    "wiki_graham": {
        "title": "Wikipedia: Donald E. Graham",
        "fetch_url": "https://en.wikipedia.org/w/index.php?title=Donald_E._Graham&action=raw",
        "cite_url": "https://en.wikipedia.org/wiki/Donald_E._Graham",
        "kind": "wikitext", "llm_person": "Donald Graham", "published": "",
    },
    "amazon_letter": {
        "title": "Amazon.com 2012 Letter to Shareholders, with the 1997 letter attached (SEC EDGAR)",
        "fetch_url": "https://www.sec.gov/Archives/edgar/data/1018724/000119312513151836/d511111dex991.htm",
        "cite_url": "https://www.sec.gov/Archives/edgar/data/1018724/000119312513151836/d511111dex991.htm",
        "kind": "html", "llm_person": "Jeff Bezos", "published": "2013-04",
    },
    "poynter_2013": {
        "title": "Poynter: Washington Post sold to Jeff Bezos (Aug 5, 2013), quoting statements by D. Graham",
        "fetch_url": "https://www.poynter.org/reporting-editing/2013/washington-post-sold-to-jeff-bezos/",
        "cite_url": "https://www.poynter.org/reporting-editing/2013/washington-post-sold-to-jeff-bezos/",
        "kind": "html", "llm_person": "Donald Graham", "published": "2013-08-05",
    },
    "wapo_10k_2012": {
        "title": "The Washington Post Company Form 10-K, fiscal year 2012 (SEC EDGAR)",
        "fetch_url": "https://www.sec.gov/Archives/edgar/data/0000104889/000010488913000009/d10k.htm",
        "cite_url": "https://www.sec.gov/Archives/edgar/data/0000104889/000010488913000009/d10k.htm",
        "kind": "html", "llm_person": "", "published": "2013-02",
    },
}

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WD_STATEMENTS_QUERY = """
SELECT ?prop ?val ?valLabel ?start ?end ?pit WHERE {{
  VALUES ?prop {{ wd:P569 wd:P19 wd:P69 wd:P108 wd:P39 wd:P166 wd:P463 wd:P22 wd:P25 }}
  wd:{qid} ?p ?st . ?prop wikibase:claim ?p ; wikibase:statementProperty ?ps . ?st ?ps ?val .
  OPTIONAL {{ ?st pq:P580 ?start }} OPTIONAL {{ ?st pq:P582 ?end }} OPTIONAL {{ ?st pq:P585 ?pit }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""
WD_FOUNDED_QUERY = """
SELECT ?org ?orgLabel ?inception WHERE {{
  ?org wdt:P112 wd:{qid} . OPTIONAL {{ ?org wdt:P571 ?inception }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


# ===========================================================================
# 通用工具
# ===========================================================================
def http_get(url: str, user_agent: str, accept: str = "*/*", retries: int = 4) -> Dict[str, Any]:
    """下载；遇到 429/503 等限流或临时错误时按 10s/20s/40s 退避重试。"""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": accept,
                                                       "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                return {"status": resp.status, "body": body.decode(charset, errors="replace"), "raw": body}
        except Exception as e:  # noqa: BLE001
            last = e
            code = getattr(e, "code", None)
            if code not in (429, 500, 502, 503, 504) and not isinstance(e, TimeoutError):
                break
            wait = 10 * (2 ** attempt)
            print(f"    retry in {wait}s ({type(e).__name__}: {e})")
            time.sleep(wait)
    raise last  # type: ignore[misc]


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return normalize_space(html.unescape(raw))


def wikitext_to_text(raw: str) -> str:
    """把维基源文本清理成纯文本：去掉模板、引用、外链、文件，保留链接文字。"""
    t = re.sub(r"(?is)<ref[^>/]*/>", " ", raw)
    t = re.sub(r"(?is)<ref[^>]*>.*?</ref>", " ", t)
    t = re.sub(r"(?s)<!--.*?-->", " ", t)
    for _ in range(10):                                    # 嵌套模板，由内向外删除
        t2 = re.sub(r"\{\{[^{}]*\}\}", " ", t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"(?is)\{\|.*?\|\}", " ", t)               # 表格
    t = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = re.sub(r"={2,}\s*([^=]+?)\s*={2,}", r". \1. ", t)
    return normalize_space(html.unescape(t))


def normalize_space(t: str) -> str:
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def canon(t: str) -> str:
    """核对“逐字引用”时用的规范化：大小写、引号、空白、首尾标点不敏感。"""
    return re.sub(r"\s+", " ", normalize_space(t).lower()).strip(" .,;:\"'")


def numbers_in(t: str) -> List[str]:
    return [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", t)]


def years_in(t: str) -> List[int]:
    return [int(y) for y in re.findall(r"\b(1[89]\d\d|20\d\d)\b", t)]


def snapshot_text(key: str) -> Optional[str]:
    path = SNAPSHOT_DIR / f"{key}.txt"
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    return wikitext_to_text(raw) if TEXT_SOURCES[key]["kind"] == "wikitext" else raw


def load_log() -> Dict[str, Any]:
    p = SNAPSHOT_DIR / "fetch_log.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ===========================================================================
# fetch：下载原始资料
# ===========================================================================
def cmd_fetch(args: argparse.Namespace) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ua = f"AIPAY-TASK research script (academic use; contact: {args.email or 'not provided'})"
    log = load_log()

    def record(key: str, url: str, res: Dict[str, Any]) -> None:
        log[key] = {"url": url, "http_status": res["status"],
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sha256": hashlib.sha256(res["raw"]).hexdigest(), "bytes": len(res["raw"])}

    for key, src in TEXT_SOURCES.items():
        try:
            res = http_get(src["fetch_url"], ua)
            text = res["body"] if src["kind"] == "wikitext" else html_to_text(res["body"])
            (SNAPSHOT_DIR / f"{key}.txt").write_text(text, encoding="utf-8")
            record(key, src["fetch_url"], res)
            print(f"[fetch] OK   {key:20s} HTTP {res['status']}  {log[key]['bytes']:>8d} bytes")
        except Exception as e:  # noqa: BLE001
            print(f"[fetch] FAIL {key:20s} {type(e).__name__}: {e}")
        time.sleep(args.sleep)

    for person, info in PEOPLE.items():
        for kind, template in (("statements", WD_STATEMENTS_QUERY), ("founded", WD_FOUNDED_QUERY)):
            key = f"wikidata_{info['qid']}_{kind}"
            query = template.format(qid=info["qid"])
            url = WIKIDATA_ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
            try:
                res = http_get(url, ua, accept="application/sparql-results+json")
                data = json.loads(res["body"])
                data["_query"] = query
                (SNAPSHOT_DIR / f"{key}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
                record(key, WIKIDATA_ENDPOINT, res)
                print(f"[fetch] OK   {key:20s} {len(data['results']['bindings'])} rows ({person})")
            except Exception as e:  # noqa: BLE001
                print(f"[fetch] FAIL {key:20s} {type(e).__name__}: {e}")
            time.sleep(args.sleep)

    (SNAPSHOT_DIR / "fetch_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


# ===========================================================================
# extract ①：Wikidata 结构化字段 → 句子模板
# ===========================================================================
def _date(v: Optional[str]) -> str:
    return (v or "")[:10]


def _fmt_day(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%B %d, %Y").replace(" 0", " ")


def wikidata_candidates() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for person, info in PEOPLE.items():
        qid = info["qid"]
        path = SNAPSHOT_DIR / f"wikidata_{qid}_statements.json"
        if not path.exists():
            print(f"[extract] wikidata snapshot missing for {person}; run fetch first")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))["results"]["bindings"]
        get = lambda r, k: r.get(k, {}).get("value")  # noqa: E731
        by_prop: Dict[str, List[Dict]] = {}
        for r in rows:
            by_prop.setdefault(get(r, "prop").rsplit("/", 1)[-1], []).append(r)

        def add(prop: str, statement: str, date: str, evidence: str, reject: str = "", cat: str = "background"):
            if not reject and re.search(r"\bQ\d+\b", statement):
                reject = "rule: Wikidata item has no English label"
            out.append({"person": person, "section": "about_person", "category": cat, "statement": statement,
                        "date": date, "source_key": f"wikidata_{qid}_statements",
                        "source_url": f"https://www.wikidata.org/wiki/{qid}#{prop}",
                        "method": "wikidata_sparql+template", "evidence": evidence, "pre_reject": reject})

        births = by_prop.get("P569", [])
        places = by_prop.get("P19", [])
        if births:
            d = _date(get(births[0], "val"))
            place = get(places[0], "valLabel") if places else ""
            add("P569", f"You are {person}, born on {_fmt_day(d)}" + (f" in {place}." if place else "."),
                d, f"P569={d}; P19={place}")
        parents = [(p, get(r, "valLabel")) for p in ("P22", "P25") for r in by_prop.get(p, [])]
        if parents:
            names = " and ".join(n for _, n in parents)
            add("P22/P25", f"Your parents are {names}.", "", f"P22/P25={names}")
        for r in by_prop.get("P69", []):
            school, s, e = get(r, "valLabel"), _date(get(r, "start")), _date(get(r, "end"))
            reject = "" if re.search(r"University|College", school or "") else "rule: only universities/colleges"
            span = f" ({s[:4]}-{e[:4]})" if s and e else (f" (graduated {e[:4]})" if e else "")
            add("P69", f"You studied at {school}{span}.", e or s, f"P69={school} {s} {e}", reject, "education")
        for prop, verb in (("P108", "worked at"), ("P463", "became a member of")):
            for r in by_prop.get(prop, []):
                org, s, e = get(r, "valLabel"), _date(get(r, "start")), _date(get(r, "end"))
                if not s:
                    add(prop, f"You {verb} {org}.", "", f"{prop}={org}", "rule: undated statement", "career")
                elif s > CUTOFF:
                    add(prop, f"You {verb} {org} in {s[:4]}.", s, f"{prop}={org} start={s}", "rule: after cutoff", "career")
                elif prop == "P108" and e and e <= CUTOFF:
                    add(prop, f"From {s[:4]} to {e[:4]} you worked at {org}.", s, f"{prop}={org} {s}..{e}", "", "career")
                elif prop == "P108":
                    add(prop, f"You have worked at {org} since {s[:4]}.", s, f"{prop}={org} start={s}", "", "career")
                else:
                    add(prop, f"In {s[:4]} you {verb} {org}.", s, f"{prop}={org} start={s}", "", "career")
        for r in by_prop.get("P39", []):
            pos, s = get(r, "valLabel"), _date(get(r, "start"))
            add("P39", f"You held the position of {pos}.", s, f"P39={pos} start={s}",
                "rule: position without organization qualifier", "career")
        for r in by_prop.get("P166", []):
            award, d = get(r, "valLabel"), _date(get(r, "pit"))
            reject = "rule: undated statement" if not d else ("rule: after cutoff" if d > CUTOFF else "")
            add("P166", f"You received the award {award}" + (f" in {d[:4]}." if d else "."), d,
                f"P166={award} pit={d}", reject, "career")

        fpath = SNAPSHOT_DIR / f"wikidata_{qid}_founded.json"
        if fpath.exists():
            for r in json.loads(fpath.read_text(encoding="utf-8"))["results"]["bindings"]:
                org, d = get(r, "orgLabel"), _date(get(r, "inception"))
                reject = "rule: undated statement" if not d else ("rule: after cutoff" if d > CUTOFF else "")
                out.append({"person": person, "section": "about_person", "category": "career",
                            "statement": f"You founded {org}" + (f" in {d[:4]}." if d else "."), "date": d,
                            "source_key": f"wikidata_{qid}_founded",
                            "source_url": get(r, "org"), "method": "wikidata_sparql+template",
                            "evidence": f"{get(r, 'org')} P112(founded by)={qid}; P571={d}", "pre_reject": reject})
    return out


# ===========================================================================
# extract ②：规则抽取 SEC 年报（2012 年 10-K）
# ===========================================================================
def regex_10k_candidates() -> List[Dict[str, Any]]:
    text = snapshot_text("wapo_10k_2012")
    if text is None:
        print("[extract] 10-K snapshot missing; run fetch first")
        return []
    src = TEXT_SOURCES["wapo_10k_2012"]
    out = []

    def add(statement: str, quote: str):
        out.append({"person": "Both", "section": "about_post", "category": "post_financials", "statement": statement,
                    "date": "2012-12-31", "source_key": "wapo_10k_2012", "source_url": src["cite_url"],
                    "method": "regex_on_sec_filing", "evidence": quote, "pre_reject": ""})

    m = re.search(r"Newspaper publishing division revenue in (\d{4}) declined (\d+)% to \$([\d.]+) million, "
                  r"from \$([\d.]+) million in (\d{4})", text)
    if m:
        y, pct, cur, prev, py = m.groups()
        add(f"Newspaper publishing division revenue fell {pct}% to ${cur} million in {y}, from ${prev} million in {py}.",
            m.group(0))
    m = re.search(r"newspaper publishing division reported an operating loss of \$([\d.]+) million in (\d{4}), "
                  r"compared to an operating loss of \$([\d.]+) million in (\d{4})", text, re.IGNORECASE)
    if m:
        cur, y, prev, py = m.groups()
        add(f"The division reported an operating loss of ${cur} million in {y}, compared with an operating loss "
            f"of ${prev} million in {py}.", m.group(0))
    m = re.search(r"Average Paid Circulation Daily Sunday((?:\s+\d{4}\s+[\d,]+\s+[\d,]+)+)", text)
    if m:
        rows = re.findall(r"(\d{4})\s+([\d,]+)\s+([\d,]+)", m.group(1))
        (y0, d0, s0), (y1, d1, s1) = rows[-2], rows[-1]
        add(f"Average paid daily circulation fell to {d1} in {y1} ({d0} in {y0}); Sunday circulation fell to "
            f"{s1} ({s0} in {y0}).", m.group(0))
    return out


# ===========================================================================
# extract ③：本地 Qwen 抽取 + 逐字引用核对
# ===========================================================================
LLM_PROMPT = """You extract verifiable facts about {person} from a source document. The facts will be used to write a
character profile for a negotiation set in July 2013.

Rules:
- Only facts about {person}: background, education, career, management philosophy, or {person}'s own stated views
  about the business. Ignore facts about other people.
- Every fact MUST include "quote": 5 to 40 words copied character-for-character from the DOCUMENT. Do not paraphrase it.
- "statement": ONE sentence in second person ("You ..." or "Your ..."), using only information in the quote.
- "year": the year the fact refers to (integer), or null.
- "category": one of background, education, career, philosophy, views_on_post.
- Return ONLY a JSON list with at most 6 items, for example:
  [{{"statement": "You ...", "quote": "...", "year": 1999, "category": "career"}}]
- If the document contains no such facts, return [].

DOCUMENT:
<<<
{chunk}
>>>"""


def chunk_text(text: str, size: int = 5000) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > size and cur:
            chunks.append(cur.strip())
            cur = ""
        cur += s + " "
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def parse_llm_json(raw: str) -> List[Dict[str, Any]]:
    m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [it for it in items if isinstance(it, dict)]


def llm_candidates(model: Any, model_name: str, reuse_cache: bool = True) -> List[Dict[str, Any]]:
    """model 需实现 generate(prompt, temperature, max_tokens)；为 None 时只读取已缓存的模型输出。"""
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for key, src in TEXT_SOURCES.items():
        person = src["llm_person"]
        text = snapshot_text(key)
        if not person or text is None:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            prompt = LLM_PROMPT.format(person=person, chunk=chunk)
            phash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            cache = LLM_DIR / f"{key}_chunk{i:02d}.json"
            cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
            if reuse_cache and cached and cached.get("prompt_sha256_16") == phash:
                raw, used_model = cached["raw_output"], cached["model"]
            elif model is not None:
                t0 = time.time()
                raw = model.generate(prompt, temperature=0.0, max_tokens=900)   # 贪心解码，可复现
                used_model = model_name
                cache.write_text(json.dumps({"source": key, "chunk": i, "model": model_name,
                                             "prompt_sha256_16": phash, "seconds": round(time.time() - t0, 1),
                                             "raw_output": raw}, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"[llm] {key} chunk {i}: {time.time() - t0:.0f}s")
            else:
                continue
            for item in parse_llm_json(raw):
                out.append({"person": person, "section": "about_person",
                            "category": str(item.get("category", "")).strip(),
                            "statement": normalize_space(str(item.get("statement", ""))),
                            "date": str(item.get("year") or ""), "source_key": key, "source_url": src["cite_url"],
                            "method": f"llm_extraction ({used_model}) + verbatim-quote check",
                            "evidence": normalize_space(str(item.get("quote", ""))), "pre_reject": "",
                            "_source_text": text, "_published": src["published"]})
    return out


def check_llm_fact(c: Dict[str, Any]) -> str:
    """返回淘汰原因；空字符串表示通过全部检查。"""
    st, quote, text = c["statement"], c["evidence"], c.pop("_source_text", "")
    if not st or not quote:
        return "llm: missing statement or quote"
    if len(quote.split()) < 4:
        return "llm: quote too short"
    if canon(quote) not in canon(text):
        return "llm: quote not found verbatim in source (possible hallucination)"
    if not re.search(r"\b(you|your)\b", " ".join(st.split()[:8]), re.IGNORECASE):   # 前 8 个词内要有 you/your
        return "llm: statement not in second person"
    if len(st.split()) > 60:
        return "llm: statement too long"
    missing = [n for n in numbers_in(st) if n not in numbers_in(quote)]
    if missing:
        return f"llm: statement contains numbers not in quote {missing}"
    if any(y > CUTOFF_YEAR for y in years_in(st) + years_in(quote)) or \
            (c["date"].isdigit() and int(c["date"]) > CUTOFF_YEAR):
        return "rule: after cutoff"
    if c["category"] not in LLM_CATEGORIES:
        return f"rule: category not allowed ({c['category']})"
    return ""


# ===========================================================================
# 人工补充条目（同样要逐字核对引用）
# ===========================================================================
def manual_candidates() -> List[Dict[str, Any]]:
    if not MANUAL_CSV.exists():
        return []
    out = []
    with open(MANUAL_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            text = snapshot_text(r["source_key"]) if r["source_key"] in TEXT_SOURCES else None
            if text is None:
                reject = "manual: source snapshot missing"
            elif canon(r["quote"]) not in canon(text):
                reject = "manual: quote not found verbatim in source"
            else:
                reject = "rule: excluded on purpose" if r["section"] == "excluded" else ""
            out.append({"person": r["person"], "section": r["section"], "category": r["category"],
                        "statement": r["statement"], "date": r.get("date", ""), "source_key": r["source_key"],
                        "source_url": TEXT_SOURCES.get(r["source_key"], {}).get("cite_url", ""),
                        "method": f"manual (reason: {r['justification']}) + verbatim-quote check",
                        "evidence": r["quote"], "pre_reject": reject})
    return out


def cmd_extract(args: argparse.Namespace) -> None:
    cands = wikidata_candidates() + regex_10k_candidates()
    model, model_name = None, ""
    if args.llm != "none":
        sys.path.insert(0, str(EXP_DIR))
        from exp_config import ExperimentConfig      # 复用实验的模型加载代码
        from model_adapter import build_model
        cfg = ExperimentConfig(backend=args.backend, model_path=args.model_path)
        model, model_name = build_model(cfg), Path(args.model_path).name
    llm = llm_candidates(model, model_name, reuse_cache=not args.no_cache)
    for c in llm:
        c["pre_reject"] = check_llm_fact(c)
    cands += llm + manual_candidates()

    for i, c in enumerate(cands, 1):
        c.pop("_source_text", None)
        c.pop("_published", None)
        c["candidate_id"] = f"C{i:03d}"
    fields = ["candidate_id", "person", "section", "category", "statement", "date", "source_key", "source_url",
              "method", "evidence", "pre_reject"]
    with open(EVIDENCE_DIR / "raw_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(cands)
    n_ok = sum(1 for c in cands if not c["pre_reject"])
    print(f"[extract] {len(cands)} candidates ({n_ok} passed per-fact checks) -> {EVIDENCE_DIR / 'raw_candidates.csv'}")


# ===========================================================================
# build：筛选规则 → CSV → prompt 文本
# ===========================================================================
def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"you", "your", "the", "a", "of", "in", "and", "to", "at"}


HEADER_PERSON = ("PUBLIC BACKGROUND ABOUT YOU (public information as of mid-2013; "
                 "use it to stay in character, do not recite it):")
HEADER_POST = ("PUBLIC FACTS ABOUT THE WASHINGTON POST "
               "(The Washington Post Company, 2012 annual report, SEC Form 10-K):")
METHOD_PRIORITY = {"wikidata": 0, "regex": 1, "manual": 2, "llm": 3}


def select(cands: List[Dict[str, str]]) -> List[Dict[str, str]]:
    log = load_log()
    for c in cands:
        reason = c["pre_reject"]
        if not reason and c["section"] != "excluded" and not c["method"].startswith("manual"):
            if OUTCOME_BLACKLIST.search(c["statement"]) or OUTCOME_BLACKLIST.search(c["evidence"]):
                reason = "rule: mentions the deal outcome/price (blacklist)"
        c["status"], c["reason"] = ("rejected", reason) if reason else ("candidate", "")
        c["retrieved_at_utc"] = log.get(c["source_key"], {}).get("fetched_at_utc", "")
        c["sha256_of_download"] = log.get(c["source_key"], {}).get("sha256", "")

    kept: Dict[str, List[Dict[str, str]]] = {}
    order = sorted((c for c in cands if c["status"] == "candidate"),
                   key=lambda c: METHOD_PRIORITY.get(c["method"].split("_")[0].split(" ")[0], 9))
    for c in order:
        group = kept.setdefault(c["person"], [])
        dup = next((g for g in group if len(_tokens(c["statement"]) & _tokens(g["statement"]))
                    / max(1, len(_tokens(c["statement"]) | _tokens(g["statement"]))) >= DEDUP_JACCARD), None)
        if dup:
            c["status"], c["reason"] = "rejected", f"rule: near-duplicate of {dup['candidate_id']}"
        elif c["section"] == "about_person" and len(group) >= MAX_FACTS_PER_PERSON:
            c["status"], c["reason"] = "rejected", f"rule: over the {MAX_FACTS_PER_PERSON}-fact limit"
        else:
            c["status"] = "selected"
            group.append(c)
    return cands


def render(person: str, selected: List[Dict[str, str]]) -> str:
    def line_order(c: Dict[str, str]) -> tuple:
        return (METHOD_PRIORITY.get(c["method"].split("_")[0].split(" ")[0], 9), c["date"] or "9999")

    mine = sorted([c for c in selected if c["person"] == person and c["section"] == "about_person"], key=line_order)
    post = [c for c in selected if c["section"] == "about_post"]
    return "\n".join([HEADER_PERSON, *[f"- {c['statement']}" for c in mine], "",
                      HEADER_POST, *[f"- {c['statement']}" for c in post]]) + "\n"


def cmd_build(args: argparse.Namespace) -> None:
    raw = EVIDENCE_DIR / "raw_candidates.csv"
    if not raw.exists():
        sys.exit("找不到 raw_candidates.csv，请先运行 extract")
    with open(raw, newline="", encoding="utf-8") as fh:
        cands = select(list(csv.DictReader(fh)))

    fields = ["candidate_id", "status", "reason", "person", "section", "category", "statement", "date",
              "method", "source_key", "source_url", "retrieved_at_utc", "sha256_of_download", "evidence"]
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cands)
    selected = [c for c in cands if c["status"] == "selected"]
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields[3:], extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)
    for person, info in PEOPLE.items():
        (PERSONA_DIR / info["file"]).write_text(render(person, selected), encoding="utf-8")
        n = sum(1 for c in selected if c["person"] == person)
        print(f"[build] {info['file']}: {n} facts about the person + "
              f"{sum(1 for c in selected if c['section'] == 'about_post')} facts about the Post")
    n_rej = sum(1 for c in cands if c["status"] == "rejected")
    print(f"[build] {len(selected)} selected, {n_rej} rejected -> {CANDIDATES_CSV.name}, {SOURCES_CSV.name}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description="Hybrid persona-fact pipeline (fetch / extract / build)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--email", default="", help="联系邮箱（SEC/Wikimedia 要求 User-Agent 中包含）")
        sp.add_argument("--sleep", type=float, default=2.0, help="两次下载之间的间隔秒数")
    for name in ("extract", "all"):
        sp = sub.choices[name] if name in sub.choices else sub.add_parser(name)
        sp.add_argument("--llm", choices=["model", "none"], default="model", help="none = 不用模型，只读缓存")
        sp.add_argument("--backend", choices=["gguf", "hf", "vllm"], default="gguf")
        sp.add_argument("--model-path", dest="model_path", default="")
        sp.add_argument("--no-cache", dest="no_cache", action="store_true", help="忽略已缓存的模型输出，重新生成")
    sub.add_parser("build")
    args = p.parse_args()
    if args.cmd in ("extract", "all") and args.llm == "model" and not args.model_path:
        p.error("extract 需要 --model-path，或用 --llm none 只使用已缓存的模型输出")
    if args.cmd in ("fetch", "all"):
        cmd_fetch(args)
    if args.cmd in ("extract", "all"):
        cmd_extract(args)
    if args.cmd in ("build", "all"):
        cmd_build(args)


if __name__ == "__main__":
    main()
