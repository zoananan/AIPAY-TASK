"""加载 AgenticPay 官方源码（不修改任何上游文件）。

third_party/AgenticPay/ 里是官方仓库 https://github.com/SafeRL-Lab/AgenticPay 在
2026-02-12 提交 1a347cf 时的 agenticpay/ 包原样拷贝（MIT 许可证，见其中 LICENSE），
来源说明见 third_party/AgenticPay/VENDORED.md。这是 arXiv v1（2026-02-05）发布后、
仓库改成多模态/合同谈判（2026-03 起）之前的最后一个版本，代码与论文描述最接近。
直接放进仓库而不是用 git submodule，是为了下载 ZIP 也能直接运行、不依赖访问 GitHub。

本文件只做一件事：把 third_party/AgenticPay 加到 sys.path，使 `import agenticpay` 可用。
"""

from __future__ import annotations

import sys

from exp_config import AGENTICPAY_PAPER_COMMIT, AGENTICPAY_ROOT

if not (AGENTICPAY_ROOT / "agenticpay" / "__init__.py").exists():
    raise ImportError(
        f"找不到 AgenticPay 源码：{AGENTICPAY_ROOT}\n"
        "third_party/AgenticPay/agenticpay/ 应随本项目一起提供；请确认项目文件完整，"
        "或设置环境变量 AGENTICPAY_ROOT 指向含有 agenticpay/ 文件夹的目录。"
    )

if str(AGENTICPAY_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTICPAY_ROOT))

UPSTREAM_COMMIT = AGENTICPAY_PAPER_COMMIT

# ---- 以下全部是上游原样代码 ----
from agenticpay import make                                               # noqa: E402  envs/registration.py
from agenticpay.agents.buyer_agent import BuyerAgent                      # noqa: E402  论文 §4.3 买方
from agenticpay.agents.seller_agent import SellerAgent                    # noqa: E402  论文 §4.3 卖方
from agenticpay.core import NegotiationStatus                             # noqa: E402
from agenticpay.models.base_llm import BaseLLM                            # noqa: E402  统一 LLM 接口

__all__ = ["make", "BuyerAgent", "SellerAgent", "NegotiationStatus", "BaseLLM", "UPSTREAM_COMMIT"]
