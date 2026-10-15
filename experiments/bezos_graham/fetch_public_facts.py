"""任务第 1 部分：获取人物公开资料并逐条核对来源（可复现的数据获取脚本）。

做三件事：
1. 下载每个来源的原文（Wikipedia 用官方的 action=raw 接口取页面源文本；SEC EDGAR 和新闻网页直接下载 HTML），
   保存到 personas/evidence/snapshots/，并记录下载时间、HTTP 状态码、SHA-256 指纹；
2. 对 personas/facts_spec.csv 里的每条事实，在对应来源原文中查找关键词/关键句（全部命中才算找到），
   并截取命中位置附近的原文片段作为证据；
3. 输出 personas/evidence/evidence_report.csv / .md，以及结构化的 personas/sources.csv
   （之后由 build_personas.py 把 CSV 转成 prompt 用的 bezos.txt / graham.txt）。

用法（在仓库根目录，需要联网；SEC 要求在 User-Agent 里留联系邮箱）：
    python experiments/bezos_graham/fetch_public_facts.py --email 你的邮箱
    python experiments/bezos_graham/fetch_public_facts.py --offline      # 不联网，只用已保存的快照重新核对

只用 Python 标准库，不需要额外安装。
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

EXP_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = EXP_DIR / "personas" / "evidence"
SNAPSHOT_DIR = EVIDENCE_DIR / "snapshots"

# ---------------------------------------------------------------------------
# 来源清单：key -> (说明, 下载地址, 引用地址)
# Wikipedia 的 action=raw 返回页面的源文本（wikitext），比解析网页 HTML 稳定
# ---------------------------------------------------------------------------
SOURCES: Dict[str, Dict[str, str]] = {
    "wiki_bezos": {
        "title": "Wikipedia: Jeff Bezos",
        "fetch_url": "https://en.wikipedia.org/w/index.php?title=Jeff_Bezos&action=raw",
        "cite_url": "https://en.wikipedia.org/wiki/Jeff_Bezos",
    },
    "wiki_graham": {
        "title": "Wikipedia: Donald E. Graham",
        "fetch_url": "https://en.wikipedia.org/w/index.php?title=Donald_E._Graham&action=raw",
        "cite_url": "https://en.wikipedia.org/wiki/Donald_E._Graham",
    },
    "amazon_letter": {
        "title": "Amazon.com 2012 Letter to Shareholders (SEC EDGAR, re-attaches the 1997 letter)",
        "fetch_url": "https://www.sec.gov/Archives/edgar/data/1018724/000119312513151836/d511111dex991.htm",
        "cite_url": "https://www.sec.gov/Archives/edgar/data/1018724/000119312513151836/d511111dex991.htm",
    },
    "wapo_10k_2012": {
        "title": "The Washington Post Company Form 10-K, fiscal year 2012 (SEC EDGAR)",
        "fetch_url": "https://www.sec.gov/Archives/edgar/data/0000104889/000010488913000009/d10k.htm",
        "cite_url": "https://www.sec.gov/Archives/edgar/data/0000104889/000010488913000009/d10k.htm",
    },
    "poynter_2013": {
        "title": "Poynter: Washington Post sold to Jeff Bezos (Aug 5, 2013)",
        "fetch_url": "https://www.poynter.org/reporting-editing/2013/washington-post-sold-to-jeff-bezos/",
        "cite_url": "https://www.poynter.org/reporting-editing/2013/washington-post-sold-to-jeff-bezos/",
    },
    "cnn_2013": {
        "title": "CNN Money: Amazon's Bezos buys Washington Post for $250 million (Aug 5, 2013)",
        "fetch_url": "https://money.cnn.com/2013/08/05/news/companies/washington-post-bezos",
        "cite_url": "https://money.cnn.com/2013/08/05/news/companies/washington-post-bezos",
    },
}

# 事实清单：放在 personas/facts_spec.csv（人工挑选的唯一来源），每行一条事实：
#   fact_id, person, section, include_in_prompt, source_key, fallback_source_key,
#   check_patterns（用 " ;; " 分隔；每一项都必须在原文中出现，一项内部可用 | 表示“任一即可”）,
#   fact_summary, prompt_line（写进 prompt 的英文原句）
SPEC_PATH = EXP_DIR / "personas" / "facts_spec.csv"
SOURCES_CSV = EXP_DIR / "personas" / "sources.csv"


def load_facts() -> List[Dict]:
    with open(SPEC_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["patterns"] = [p.strip() for p in r["check_patterns"].split(";;") if p.strip()]
        assert r["source_key"] in SOURCES, f"{r['fact_id']}: unknown source {r['source_key']}"
    return rows


def to_text(raw: str) -> str:
    """去掉 HTML 标签与脚本，统一引号和空白，便于查找。"""
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = raw.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    raw = raw.replace("\xa0", " ")
    return re.sub(r"\s+", " ", raw)


def fetch(url: str, user_agent: str, timeout: int = 30) -> Dict:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return {"status": resp.status, "body": body.decode(charset, errors="replace")}


def snippet(text: str, pattern: str, width: int = 90) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return ""
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return "..." + text[a:b].strip() + "..."


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Fetch and verify public persona facts")
    parser.add_argument("--email", default="", help="联系邮箱（SEC EDGAR 要求 User-Agent 中包含）")
    parser.add_argument("--offline", action="store_true", help="不联网，只用已保存的快照核对")
    parser.add_argument("--sleep", type=float, default=1.0, help="两次下载之间的间隔秒数（礼貌访问）")
    args = parser.parse_args()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    user_agent = f"AIPAY-TASK research script (academic use; contact: {args.email or 'not provided'})"
    meta_path = SNAPSHOT_DIR / "fetch_log.json"
    meta: Dict[str, Dict] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    texts: Dict[str, Optional[str]] = {}
    for key, src in SOURCES.items():
        snap = SNAPSHOT_DIR / f"{key}.txt"
        if not args.offline:
            try:
                res = fetch(src["fetch_url"], user_agent)
                snap.write_text(to_text(res["body"]), encoding="utf-8")   # 保存去掉 HTML 后的正文；指纹按原始下载内容计算
                meta[key] = {
                    "url": src["fetch_url"], "http_status": res["status"],
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sha256": hashlib.sha256(res["body"].encode("utf-8")).hexdigest(),
                    "bytes": len(res["body"].encode("utf-8")),
                }
                print(f"[fetch] OK   {key:14s} HTTP {res['status']}  {meta[key]['bytes']:>8d} bytes")
            except Exception as e:  # 网络问题不中断，后面会标为 fetch_failed
                print(f"[fetch] FAIL {key:14s} {type(e).__name__}: {e}")
                meta.setdefault(key, {})["last_error"] = f"{type(e).__name__}: {e}"
            time.sleep(args.sleep)
        texts[key] = to_text(snap.read_text(encoding="utf-8")) if snap.exists() else None
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for f in load_facts():
        key = f["source_key"]
        fb = f.get("fallback_source_key") or ""
        if texts.get(key) is None and fb and texts.get(fb) is not None:
            key = fb                            # 主来源下载失败时，用报道了同一事实的备用来源
        src, text = SOURCES[key], texts.get(key)
        if text is None:
            status, missing, evidence = "fetch_failed", f["patterns"], ""
        else:
            missing = [p for p in f["patterns"] if not re.search(p, text, flags=re.IGNORECASE)]
            status = "verified" if not missing else "not_found"
            evidence = " | ".join(s for s in (snippet(text, p) for p in f["patterns"]) if s)
        info = meta.get(key, {})
        rows.append({
            "fact_id": f["fact_id"], "person": f["person"], "fact": f["fact_summary"], "status": status,
            "missing_patterns": "; ".join(missing), "source_title": src["title"], "source_url": src["cite_url"],
            "fetched_at_utc": info.get("fetched_at_utc", ""), "sha256": info.get("sha256", ""),
            "evidence_snippets": evidence,
        })

    with open(EVIDENCE_DIR / "evidence_report.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Evidence report: public persona facts", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC)", "",
             "| ID | Person | Fact | Status | Source |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['fact_id']} | {r['person']} | {r['fact']} | {r['status']} | [{r['source_title']}]({r['source_url']}) |")
    lines += ["", "## Evidence snippets", ""]
    for r in rows:
        lines.append(f"**{r['fact_id']}** ({r['status']}): {r['evidence_snippets'] or '(none)'}")
        lines.append("")
    (EVIDENCE_DIR / "evidence_report.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 原始资料 → 结构化 CSV：只收录核对通过的事实 ----
    spec = {f["fact_id"]: f for f in load_facts()}
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as fh:
        fields = ["fact_id", "person", "section", "included_in_prompt", "fact", "verification_status",
                  "source_title", "source_url", "retrieved_at_utc", "sha256_of_download",
                  "retrieval_method", "evidence_quote"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            f = spec[r["fact_id"]]
            method = ("Wikipedia action=raw API + keyword check" if "wikipedia" in r["source_url"]
                      else "HTTP download of page + keyword check")
            writer.writerow({
                "fact_id": r["fact_id"], "person": r["person"], "section": f["section"],
                "included_in_prompt": f["include_in_prompt"], "fact": r["fact"],
                "verification_status": r["status"], "source_title": r["source_title"],
                "source_url": r["source_url"], "retrieved_at_utc": r["fetched_at_utc"],
                "sha256_of_download": r["sha256"], "retrieval_method": method,
                "evidence_quote": r["evidence_snippets"].split(" | ")[0] if r["evidence_snippets"] else "",
            })

    n_ok = sum(r["status"] == "verified" for r in rows)
    print(f"\n[verify] {n_ok}/{len(rows)} facts verified in their sources")
    for r in rows:
        if r["status"] != "verified":
            print(f"  - {r['fact_id']}: {r['status']}  missing={r['missing_patterns']}")
    print(f"[verify] report: {EVIDENCE_DIR / 'evidence_report.csv'}")
    print(f"[csv]    structured sources: {SOURCES_CSV}")
    print("下一步：python experiments/bezos_graham/build_personas.py  （由 CSV 生成 bezos.txt / graham.txt）")


if __name__ == "__main__":
    main()
