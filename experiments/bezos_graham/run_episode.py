"""只跑一次谈判（第一次运行时用它检查模型和 prompt 是否正常）。

用法（在仓库根目录）：
    python experiments/bezos_graham/run_episode.py                     # Qwen3-4B，本地离线
    python experiments/bezos_graham/run_episode.py --backend mock      # 不加载模型，几秒钟跑通流程
    python experiments/bezos_graham/run_episode.py --show-prompt       # 额外打印第 1 轮双方的完整 prompt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from exp_config import ExperimentConfig, add_common_cli_args, apply_cli_overrides
from episode import run_episode
from model_adapter import build_model


def main() -> None:
    # 上游 render() 会打印 ✓/✗；Windows 中文控制台在输出被重定向时可能无法编码，替换掉而不是报错
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Bezos vs Graham: one AgenticPay negotiation")
    parser.add_argument("--run-id", dest="run_id", type=int, default=0, help="seed = base_seed + run_id")
    parser.add_argument("--show-prompt", dest="show_prompt", action="store_true")
    add_common_cli_args(parser)
    args = parser.parse_args()

    cfg = apply_cli_overrides(ExperimentConfig(), args)
    out_dir = Path(cfg.output_dir) / f"single_{cfg.backend}_{datetime.now():%Y%m%d_%H%M%S}"
    model = build_model(cfg)
    rec = run_episode(cfg, model, args.run_id, out_dir)

    if args.show_prompt:
        for role in ("buyer", "seller"):
            print(f"\n{'#' * 30} Round-1 {role.upper()} PROMPT {'#' * 30}\n{rec['first_prompts'][role]}")

    print("\n" + "=" * 60)
    print(f"status       : {rec['status']}")
    print(f"agreed price : {rec['agreed_price']}  (USD millions; inside [200, 350]: {rec['deal_in_zone']})")
    print(f"rounds       : {rec['rounds']}")
    print(f"last offers  : buyer={rec['final_buyer_offer']}  seller={rec['final_seller_offer']}")
    print(f"scores       : upstream={rec['upstream_scores']}")
    print(f"saved to     : {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
