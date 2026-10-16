"""分析每次运行是怎么成交（或为什么没成交）的，用来评估买方“截止轮”这个变量。

用法（在仓库根目录）：
    python experiments/bezos_graham/analyze_closing.py experiments/bezos_graham/outputs/A1_disciplined_gguf_XXXX
    python experiments/bezos_graham/analyze_closing.py outputs/A1_* outputs/A1d12_* outputs/A1nod_*   # 多组对比

对每次成交，判断是谁“报出对方的数字”：
    buyer_copied_seller   买方照抄了卖方上一轮的报价（v3 的 ACCEPT 步骤）
    seller_copied_buyer   卖方照抄了买方本轮的报价
    other                 其他（双方同时报出同一个新数字）
对买方照抄的情况，再判断触发了哪条 ACCEPT 条件（可能同时满足多条）：
    small_gap       买方发言时双方差距 <= 10（v5 的接受条件）
    deadline        已到买方 prompt 里的截止轮
    seller_stalled  卖方最近两条消息报价相同
    ignored_close   买方上一轮提出了收尾价（差距 <= 10），卖方没有照写
并记录接受时卖方是否“还在降价”（上一轮比再上一轮低）以及降了多少：
如果截止轮触发时卖方还在降价，说明截止规则可能让买方放弃了后续的降价空间。

输出：屏幕上的逐次运行表格，以及每个结果目录下的 closing_analysis.csv。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

TAG_ONLY_RE = re.compile(r"\s*\**\s*###\s*BUYER_PRICE\(\$[\d.,]+\)\s*###\s*\**\s*")
# v4 写法 “..., or if current_round ... is 7 or more”；v3 写法 “if current_round ... is 7 or more, OR ...”
DEADLINE_RE = re.compile(r"or if current_round in the negotiation state is (\d+) or more"
                         r"|if current_round in the negotiation state is (\d+) or more, OR")


def offers(record: Dict[str, Any], role: str) -> Dict[int, Optional[float]]:
    return {t["round"]: t["env_parsed_price"] for t in record["turns"] if t["role"] == role}


def deadline_round(record: Dict[str, Any]) -> Optional[int]:
    """从买方实际收到的 prompt 中读出截止轮（1 起算）；没有截止条款时返回 None。"""
    m = DEADLINE_RE.search(record.get("first_prompts", {}).get("buyer", ""))
    return int(m.group(1) or m.group(2)) + 1 if m else None          # state 里的 current_round 从 0 起算


def analyze_run(record: Dict[str, Any]) -> Dict[str, Any]:
    b, s = offers(record, "buyer"), offers(record, "seller")
    dl = deadline_round(record)
    row: Dict[str, Any] = {"run": record["run_idx"], "status": record["status"], "rounds": record["rounds"],
                           "price": record["agreed_price"], "deadline_round": dl or "none",
                           "closed_by": "", "triggers": "", "seller_still_lowering": "", "seller_last_drop": "",
                           # 买方只发了价格标签、没有任何文字的消息数（v3 试运行中的主要失败模式）
                           "buyer_tag_only": sum(1 for x in record["turns"] if x["role"] == "buyer"
                                                 and TAG_ONLY_RE.fullmatch(x.get("text", "")))}
    if not record["deal"]:
        lb, ls = b.get(record["rounds"]), s.get(record["rounds"])
        row["closed_by"] = "timeout"
        row["triggers"] = f"final gap {ls - lb:g}" if lb is not None and ls is not None else ""
        return row

    t = record["rounds"]
    if t >= 2 and b.get(t) is not None and b.get(t) == s.get(t - 1):
        row["closed_by"] = "buyer_copied_seller"
        trig = []
        if dl is not None and t >= dl:
            trig.append("deadline")
        # v5 的接受条件：买方发言时 state 里的 seller_price（第 t-1 轮卖方报价）减 buyer_price（第 t-1 轮买方报价）<= 10
        if None not in (s.get(t - 1), b.get(t - 1)) and s[t - 1] - b[t - 1] <= 10:
            trig.append("small_gap")
        if t >= 3 and s.get(t - 1) is not None and s.get(t - 1) == s.get(t - 2):
            trig.append("seller_stalled")
        # 收尾步骤在第 t-1 轮执行的条件：当时卖方最新报价（第 t-2 轮）比买方上一次报价（第 t-2 轮）高不超过 10
        if t >= 3 and None not in (b.get(t - 2), s.get(t - 2), s.get(t - 1)) and s[t - 2] - b[t - 2] <= 10 \
                and s.get(t - 1) != b.get(t - 1):
            trig.append("ignored_close")
        row["triggers"] = "+".join(trig) or "other"
        if t >= 3 and None not in (s.get(t - 1), s.get(t - 2)):
            drop = s[t - 2] - s[t - 1]
            row["seller_still_lowering"] = drop > 0
            row["seller_last_drop"] = round(drop, 2)
    elif b.get(t) is not None and s.get(t) == b.get(t):
        row["closed_by"] = "seller_copied_buyer"
    else:
        row["closed_by"] = "other"
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dirs", nargs="+", help="run_experiments.py 生成的结果目录")
    args = p.parse_args()
    fields = ["run", "status", "rounds", "price", "deadline_round", "closed_by", "triggers",
              "seller_still_lowering", "seller_last_drop", "buyer_tag_only"]
    for d in map(Path, args.dirs):
        runs = sorted(d.glob("run_[0-9][0-9].json"))
        if not runs:
            print(f"{d}: no run_XX.json found")
            continue
        rows: List[Dict[str, Any]] = [analyze_run(json.loads(r.read_text(encoding="utf-8"))) for r in runs]
        with open(d / "closing_analysis.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\n=== {d.name}  (buyer deadline round: {rows[0]['deadline_round']}) ===")
        print(" | ".join(f"{f:>12s}" for f in fields))
        for r in rows:
            print(" | ".join(f"{str(r[f]):>12s}" for f in fields))
        deals = [r for r in rows if r["status"] == "agreed"]
        dl_hits = [r for r in deals if "deadline" in str(r["triggers"])]
        dl_moving = [r for r in dl_hits if r["seller_still_lowering"] is True]
        print(f"deals {len(deals)}/{len(rows)}; closed by buyer copying seller: "
              f"{sum(r['closed_by'] == 'buyer_copied_seller' for r in deals)}; "
              f"deadline involved: {len(dl_hits)}; deadline while seller still lowering: {len(dl_moving)}")


if __name__ == "__main__":
    main()
