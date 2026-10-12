"""批量运行：同一配置跑 N 次独立谈判（默认 10 次），输出任务要求的三项指标。

用法（在仓库根目录）：
    python experiments/bezos_graham/run_experiments.py --runs 10 --tag A0_paper_default
    python experiments/bezos_graham/run_experiments.py --backend mock --runs 3      # 无模型自检

输出目录 experiments/bezos_graham/outputs/<tag>_<时间戳>/ 下：
    config.json              本次实验的完整配置（便于复现）
    run_XX.json / run_XX_transcript.txt   每次运行的完整对话、逐条解析结果、评分
    runs.csv                 每次运行一行
    summary.json             汇总：平均成交价、成交率、平均轮数等
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from exp_config import ExperimentConfig, add_common_cli_args, apply_cli_overrides
from episode import run_episode
from metrics import aggregate
from model_adapter import build_model
from upstream import UPSTREAM_COMMIT

CSV_FIELDS = ["run_idx", "seed", "status", "deal", "agreed_price", "deal_in_zone", "rounds",
              "final_buyer_offer", "final_seller_offer", "tag_compliance", "elapsed_s"]


def print_summary(tag: str, s: dict) -> None:
    def fmt(x, nd=2):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    print("\n" + "=" * 64)
    print(f"Summary [{tag}]  (prices in USD millions)")
    print("=" * 64)
    print(f"  runs                     : {s['n_runs']}")
    print(f"  deals (deal rate)        : {s['n_deals']} ({fmt(100 * s['deal_rate'] if s['deal_rate'] is not None else None, 0)}%)")
    print(f"  deals inside [min, max]  : {s['n_deals_in_zone']}")
    print(f"  avg final price (deals)  : {fmt(s['avg_final_price'])}  ± {fmt(s['std_final_price'])}"
          f"  [min {fmt(s['min_final_price'])}, max {fmt(s['max_final_price'])}]")
    print(f"  avg rounds per run       : {fmt(s['avg_rounds'])}   (deals only: {fmt(s['avg_rounds_deals_only'])})")
    print(f"  timeout rate             : {fmt(s['timeout_rate'])}")
    print(f"  price-tag compliance     : {fmt(s['tag_compliance'])}")
    print("=" * 64)


def main(argv: Optional[List[str]] = None, default_runs: int = 10) -> dict:
    # 上游 render() 会打印 ✓/✗；Windows 中文控制台在输出被重定向时可能无法编码，替换掉而不是报错
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Bezos vs Graham: N independent AgenticPay negotiations")
    parser.add_argument("--runs", type=int, default=default_runs, help="独立运行次数（默认 10）")
    parser.add_argument("--start-run", dest="start_run", type=int, default=0,
                        help="从第几个 run 开始（中断后续跑时用；seed = base_seed + run_idx）")
    parser.add_argument("--tag", default="A0_paper_default", help="本组实验的名字（如 A0/A1/A2）")
    add_common_cli_args(parser)
    args = parser.parse_args(argv)

    cfg = apply_cli_overrides(ExperimentConfig(), args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / f"{args.tag}_{cfg.backend}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"tag": args.tag, "agenticpay_commit": UPSTREAM_COMMIT, **cfg.to_dict()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[exp] tag={args.tag} backend={cfg.backend} runs={args.runs} -> {out_dir}")
    print(f"[exp] AgenticPay commit {UPSTREAM_COMMIT}; temperature={cfg.temperature}; "
          f"max_rounds={cfg.max_rounds}; buyer_max={cfg.buyer_max_price:g}; seller_min={cfg.seller_min_price:g}")

    model = build_model(cfg)   # 只加载一次，所有运行、买卖双方共用
    records = []
    with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run_idx in range(args.start_run, args.start_run + args.runs):
            rec = run_episode(cfg, model, run_idx, out_dir)
            records.append(rec)
            tags = [t["audit"]["tag_ok"] for t in rec["turns"]]
            row = {k: rec.get(k) for k in CSV_FIELDS}
            row["tag_compliance"] = round(sum(tags) / len(tags), 3) if tags else None
            writer.writerow(row)
            fcsv.flush()       # 每跑完一次就落盘，中途中断也不丢数据
            print(f"[exp] run {run_idx}: status={rec['status']} price={rec['agreed_price']} "
                  f"rounds={rec['rounds']} ({rec['elapsed_s']}s)")

    summary = aggregate(records)
    summary["tag"] = args.tag
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(args.tag, summary)
    print(f"[exp] results saved to {out_dir}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
