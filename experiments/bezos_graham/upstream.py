"""加载 AgenticPay 官方源码（不修改任何上游文件）。

我们把官方仓库作为 git submodule 放在 third_party/AgenticPay，并固定在
2026-02-12 的提交 1a347cf —— 这是 arXiv v1（2026-02-05）发布后、仓库改成多模态/
合同谈判（2026-03 起）之前的最后一个版本，代码与论文描述最接近。

本文件只做两件事：
1. 把 third_party/AgenticPay 加到 sys.path，使 `import agenticpay` 可用；
2. 检查 submodule 是否真的停在论文版本，不是的话给出警告。
"""

from __future__ import annotations

import subprocess
import sys
import warnings

from exp_config import AGENTICPAY_PAPER_COMMIT, AGENTICPAY_ROOT


def _check_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(AGENTICPAY_ROOT), "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if head and head != AGENTICPAY_PAPER_COMMIT:
        warnings.warn(
            f"third_party/AgenticPay 当前在 {head}，不是论文版本 {AGENTICPAY_PAPER_COMMIT}。"
            f"请运行: git -C third_party/AgenticPay checkout {AGENTICPAY_PAPER_COMMIT}"
        )
    return head or "unknown"


if not (AGENTICPAY_ROOT / "agenticpay" / "__init__.py").exists():
    raise ImportError(
        f"找不到 AgenticPay 源码：{AGENTICPAY_ROOT}\n"
        "请在仓库根目录运行：git submodule update --init\n"
        "或手动：git clone https://github.com/SafeRL-Lab/AgenticPay.git third_party/AgenticPay "
        f"&& git -C third_party/AgenticPay checkout {AGENTICPAY_PAPER_COMMIT}"
    )

if str(AGENTICPAY_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTICPAY_ROOT))

UPSTREAM_COMMIT = _check_commit()

# ---- 以下全部是上游原样代码 ----
from agenticpay import make                                               # noqa: E402  envs/registration.py
from agenticpay.agents.buyer_agent import BuyerAgent                      # noqa: E402  论文 §4.3 买方
from agenticpay.agents.seller_agent import SellerAgent                    # noqa: E402  论文 §4.3 卖方
from agenticpay.core import NegotiationStatus                             # noqa: E402
from agenticpay.models.base_llm import BaseLLM                            # noqa: E402  统一 LLM 接口

__all__ = ["make", "BuyerAgent", "SellerAgent", "NegotiationStatus", "BaseLLM", "UPSTREAM_COMMIT"]
