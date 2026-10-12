"""实验配置：Bezos（买方）收购《华盛顿邮报》，Graham（卖方）出售。

所有“可调参数”集中在这里，命令行参数（见 run_episode.py / run_experiments.py）
只是对这里默认值的覆盖。每个参数都标注了来源：
    [PAPER]    论文正文/附录给出的设定
    [UPSTREAM] 论文对应版本源码（AgenticPay @ 1a347cf）中的默认值
    [TASK]     教授任务书指定
    [OURS]     我们为本任务新增/选择的设定（需在报告中说明）
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parent              # experiments/bezos_graham/
REPO_ROOT = EXP_DIR.parent.parent                       # 仓库根目录
PROMPT_DIR = EXP_DIR / "prompts"

# AgenticPay 官方源码位置（git submodule，固定在论文对应提交 1a347cf）。
# 如放在别处，可设置环境变量 AGENTICPAY_ROOT 覆盖。
AGENTICPAY_ROOT = Path(os.environ.get("AGENTICPAY_ROOT", REPO_ROOT / "third_party" / "AgenticPay"))
AGENTICPAY_PAPER_COMMIT = "1a347cf"

# 本地模型目录（先运行 download_model.py 下载一次，之后完全离线）。
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"        # [TASK]
DEFAULT_MODEL_PATH = Path(os.environ.get("QWEN_MODEL_PATH", REPO_ROOT / "models" / "Qwen3-4B-Instruct-2507"))

DEFAULT_OUTPUT_DIR = EXP_DIR / "outputs"


@dataclass
class ExperimentConfig:
    # ---------------- 价格设定 ----------------
    buyer_max_price: float = 350.0          # [TASK] 买方秘密上限（单位：百万美元）
    seller_min_price: float = 200.0         # [TASK] 卖方秘密底价（单位：百万美元）
    # Table 15 模板里有 "Your initial asking price is ${initial_price}"，任务书没给，
    # 这里取 400：高于买方上限，仿照论文 Business Acquisition 示例（要价 150k > 买方上限 120k）。
    initial_asking_price: float = 400.0     # [OURS] 可用 --initial-price 修改

    # ---------------- 环境规则 ----------------
    max_rounds: int = 20                    # [PAPER §5.1] 最多 20 轮（1 轮 = 买方 1 句 + 卖方 1 句）
    # 论文 §4.1：双方报出相同价格才算成交；论文版 examples/config.py 中 price_tolerance = 0.0
    price_tolerance: float = 0.0            # [UPSTREAM config.py]
    # 上游 step() 在递增轮次前判断 current_round >= max_rounds，实际允许 max_rounds+1 轮。
    # True：给环境传 max_rounds-1，使对话恰好最多 20 轮（与论文表述及 Table 20 一致）。
    fix_round_cap: bool = True              # [OURS]

    # 论文 §5.1 / Algorithm 1 的评分参数（也正是 1a347cf 版环境的默认值）
    gamma: float = 0.99
    deal_weight_D: float = 30.0
    quality_weight_W: float = 55.0
    efficiency_weight_E: float = 15.0
    failure_penalty_F: float = 15.0

    # ---------------- 生成参数 ----------------
    # 论文 §5.1：temperature 0、seed 0、每次最多生成 1024 tokens（确定性解码）。
    # 但任务要求“10 次独立运行”，确定性解码会得到 10 次完全相同的对话，
    # 所以默认用 Qwen3-4B-Instruct-2507 官方推荐的采样参数，每次运行换一个 seed。
    # 想复现论文的确定性协议：命令行加 --greedy。
    temperature: float = 0.7                # [OURS，取自 Qwen 模型卡推荐值]
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 1024              # [PAPER §5.1]
    base_seed: int = 0                      # 第 i 次运行使用 seed = base_seed + i

    # ---------------- 角色名 ----------------
    # Table 14/15 中写的是 "You are Buyer" / "You are Seller"，名字来自上游 Agent 的 name 参数。
    # 我们把它设成真实人物名；改回 "Buyer"/"Seller" 即与论文表格逐字一致。
    buyer_name: str = "Jeff Bezos"          # [TASK]
    seller_name: str = "Donald Graham"      # [TASK]

    # ---------------- Prompt 模板 ----------------
    buyer_template: str = "buyer_table14.txt"   # [PAPER Table 14]
    seller_template: str = "seller_table15.txt" # [PAPER Table 15]
    # 下一步（加入公开人物资料）预留的位置：非空时插在模板最后一行 "Now, respond as ..." 之前。
    buyer_persona: str = ""
    seller_persona: str = ""

    # ---------------- 场景信息（双方都能看到的公开信息） ----------------
    # 买方的 user_requirement（只进入买方 context，对应上游 Task1 示例中的同名参数）
    user_requirement: str = "Acquire The Washington Post newspaper business."
    # 商品信息 v_j（论文 §3 Product and Market Context），双方都可见。
    # 注意：这里绝不能写 350 或 200，否则破坏不完全信息设定。
    product_info: Dict[str, Any] = field(default_factory=lambda: {
        "name": "The Washington Post",
        "category": "Business acquisition (newspaper publishing business)",
        "transaction": "Sale of The Washington Post newspaper business as a whole",
        "price_unit": ("All prices in this negotiation, including the values inside the price tags, "
                       "are in millions of US dollars: $X means X million USD."),
    })
    # 市场/环境信息 x（论文 §3 shared market context），双方都可见。
    # 这一步只放双方姓名，不放任何人物背景事实——那是下一步（联网检索公开资料）的内容。
    environment_info: Dict[str, Any] = field(default_factory=lambda: {
        "deal_type": "One-to-one acquisition negotiation",
        "buyer": "Jeff Bezos",
        "seller": "Donald Graham",
    })

    # ---------------- 模型/运行 ----------------
    backend: str = "hf"                     # "hf"（transformers，Win/Mac/Linux 通用）| "vllm"（仅 Linux+GPU）| "mock"（无模型自检）
    model_path: str = str(DEFAULT_MODEL_PATH)
    device: str = "auto"                    # auto | cuda | mps | cpu
    offline: bool = True                    # True：禁止联网，只从本地目录加载模型
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    verbose: bool = True                    # 是否在终端打印每轮对话

    # ------------------------------------------------------------------
    def env_max_rounds(self) -> int:
        """真正传给上游环境的 max_rounds（见 fix_round_cap 的说明）。"""
        return self.max_rounds - 1 if self.fix_round_cap else self.max_rounds

    def sampling_kwargs(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_new_tokens,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_cli_overrides(cfg: ExperimentConfig, args: Any) -> ExperimentConfig:
    """把 argparse 的结果覆盖到配置上（只覆盖用户显式给出的参数）。"""
    mapping = {
        "backend": "backend",
        "model_path": "model_path",
        "device": "device",
        "initial_price": "initial_asking_price",
        "max_rounds": "max_rounds",
        "base_seed": "base_seed",
        "temperature": "temperature",
        "output_dir": "output_dir",
        "buyer_template": "buyer_template",
        "seller_template": "seller_template",
    }
    for arg_name, cfg_name in mapping.items():
        value: Optional[Any] = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg, cfg_name, value)
    if getattr(args, "greedy", False):
        # 论文 §5.1 的确定性解码协议
        cfg.temperature = 0.0
    if getattr(args, "paper_names", False):
        cfg.buyer_name, cfg.seller_name = "Buyer", "Seller"
    if getattr(args, "no_round_fix", False):
        cfg.fix_round_cap = False
    if getattr(args, "online", False):
        cfg.offline = False
    if getattr(args, "quiet", False):
        cfg.verbose = False
    return cfg


def add_common_cli_args(parser: Any) -> None:
    parser.add_argument("--backend", choices=["hf", "vllm", "mock"], help="推理后端（默认 hf）")
    parser.add_argument("--model-path", dest="model_path", help="本地模型目录")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--initial-price", dest="initial_price", type=float, help="卖方初始要价（百万美元）")
    parser.add_argument("--max-rounds", dest="max_rounds", type=int)
    parser.add_argument("--seed", dest="base_seed", type=int, help="基础随机种子")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--greedy", action="store_true", help="论文协议：temperature=0 确定性解码")
    parser.add_argument("--paper-names", dest="paper_names", action="store_true",
                        help="角色名用 Buyer/Seller（与 Table 14/15 逐字一致）")
    parser.add_argument("--no-round-fix", dest="no_round_fix", action="store_true",
                        help="保留上游 max_rounds+1 的轮数行为")
    parser.add_argument("--buyer-template", dest="buyer_template", help="prompts/ 下的买方模板文件名")
    parser.add_argument("--seller-template", dest="seller_template", help="prompts/ 下的卖方模板文件名")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--online", action="store_true", help="允许联网加载模型（默认离线）")
    parser.add_argument("--quiet", action="store_true", help="不打印每轮对话")
