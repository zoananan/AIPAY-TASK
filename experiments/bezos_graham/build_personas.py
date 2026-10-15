"""任务第 1 部分最后一步：把结构化 CSV 转成写进 prompt 的人物资料文本。

输入：
    personas/facts_spec.csv   每条事实写进 prompt 的英文原句（prompt_line）、所属人物、所属段落
    personas/sources.csv      fetch_public_facts.py 生成的核对结果（verification_status）
输出：
    personas/bezos.txt        买方资料：Bezos 本人的事实 + 《邮报》公开事实
    personas/graham.txt       卖方资料：Graham 本人的事实 + 《邮报》公开事实

规则：只收录 include_in_prompt = yes 且在 sources.csv 中 verification_status = verified 的事实；
没核对通过的事实会被跳过并打印警告，保证 prompt 里的每一句都有来源可查。

用法（在仓库根目录）：
    python experiments/bezos_graham/fetch_public_facts.py --offline    # 先生成/更新 sources.csv
    python experiments/bezos_graham/build_personas.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

PERSONA_DIR = Path(__file__).resolve().parent / "personas"
SPEC_PATH = PERSONA_DIR / "facts_spec.csv"
SOURCES_CSV = PERSONA_DIR / "sources.csv"

HEADER_PERSON = ("PUBLIC BACKGROUND ABOUT YOU (public information as of mid-2013; "
                 "use it to stay in character, do not recite it):")
HEADER_POST = ("PUBLIC FACTS ABOUT THE WASHINGTON POST "
               "(The Washington Post Company, 2012 annual report, SEC Form 10-K):")

OUTPUTS = {"Jeff Bezos": "bezos.txt", "Donald Graham": "graham.txt"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build(person: str, spec: List[Dict[str, str]], verified: Dict[str, str]) -> str:
    def lines(section: str, who: str) -> List[str]:
        out = []
        for r in spec:
            if r["section"] != section or r["person"] != who or r["include_in_prompt"] != "yes":
                continue
            status = verified.get(r["fact_id"], "missing")
            if status != "verified":
                print(f"  [skip] {r['fact_id']} ({who}): verification_status={status}")
                continue
            out.append(f"- {r['prompt_line']}")
        return out

    return "\n".join([HEADER_PERSON, *lines("about_person", person), "",
                      HEADER_POST, *lines("about_post", "Both")]) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if not SOURCES_CSV.exists():
        sys.exit("找不到 personas/sources.csv，请先运行 fetch_public_facts.py")
    spec = read_csv(SPEC_PATH)
    verified = {r["fact_id"]: r["verification_status"] for r in read_csv(SOURCES_CSV)}
    for person, filename in OUTPUTS.items():
        text = build(person, spec, verified)
        (PERSONA_DIR / filename).write_text(text, encoding="utf-8")
        n = sum(1 for line in text.splitlines() if line.startswith("- "))
        print(f"[build] {filename}: {n} facts")


if __name__ == "__main__":
    main()
