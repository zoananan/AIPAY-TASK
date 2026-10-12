"""逐条消息的格式审计、论文 Algorithm 1 评分、多次运行的汇总统计。"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

# 与上游 Task1BasicPriceNegotiation._extract_price 第 1 优先级的正则相同
STRICT_TAG = re.compile(r"###\s*(BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###", re.IGNORECASE)


def audit_message(text: str, role: str, own_reservation: float) -> Dict[str, Any]:
    """检查一条消息是否遵守 Table 14/15 的格式要求。

    上游解析器在找不到标准标签时会退回去抓文中任意 "$数字"，可能抓错价格，
    所以我们额外记录“是否恰好有一个本方标准标签”，用来判断这次解析是否可信。
    """
    tags = STRICT_TAG.findall(text)
    expected = "BUYER_PRICE" if role == "buyer" else "SELLER_PRICE"
    own_tags = [t for t in tags if t[0].upper() == expected]
    strict_price = float(own_tags[-1][1].replace(",", "")) if own_tags else None

    # 粗略的“泄露保留价”检查：把价格标签去掉后，正文里是否出现了自己的保留价数字
    body = STRICT_TAG.sub(" ", text)
    res = f"{own_reservation:g}"
    possible_leak = re.search(rf"(?<![\d.]){re.escape(res)}(?:\.0+)?(?![\d])", body) is not None

    return {
        "n_tags": len(tags),
        "tag_ok": len(tags) == 1 and len(own_tags) == 1,
        "strict_price": strict_price,
        "make_deal": "MAKE_DEAL" in text.upper(),
        "possible_leak": possible_leak,
    }


def paper_algorithm1_scores(price: Optional[float], deal_round: int, agreed: bool,
                            p_max: float, p_min: float, max_rounds: int,
                            D: float = 30.0, W: float = 55.0, E: float = 15.0,
                            F: float = 15.0, gamma: float = 0.99) -> Dict[str, float]:
    """论文 Algorithm 1 的逐行实现（t 为 1 起算的成交轮次）。

    上游 1a347cf 的实现里 discount 用的是 γ^t 而不是 γ^(t-1)（成交后先把轮次 +1 再算分），
    所以上游分数会比论文公式略低；两者都保存在结果里。
    """
    Z = p_max - p_min
    if agreed and price is not None and p_min <= price <= p_max:
        d = gamma ** (deal_round - 1)
        r_b = (p_max - price) / Z
        r_s = (price - p_min) / Z
        Q = 4 * r_b * r_s
        return {
            "global": d * (D + Q * W + E),
            "buyer": d * (D + r_b * W + E),
            "seller": d * (D + r_s * W + E),
        }
    d = gamma ** (max_rounds - 1)
    penalty = -F * (1 - d)
    return {"global": penalty, "buyer": penalty, "seller": penalty}


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """任务要求的三个指标 + 若干辅助指标。

    - avg_final_price：只在成交的运行上平均（超时没有成交价）；
    - deal_rate：成交次数 / 运行次数；
    - avg_rounds：所有运行的平均轮数（超时记为最大轮数）。
    """
    n = len(runs)
    deals = [r for r in runs if r["deal"]]
    prices = [r["agreed_price"] for r in deals]
    valid = [r for r in deals if r["deal_in_zone"]]
    all_tags = [t["audit"]["tag_ok"] for r in runs for t in r["turns"]]
    return {
        "n_runs": n,
        "n_deals": len(deals),
        "deal_rate": len(deals) / n if n else None,
        "n_deals_in_zone": len(valid),
        "avg_final_price": _mean(prices),
        "std_final_price": _std(prices),
        "min_final_price": min(prices) if prices else None,
        "max_final_price": max(prices) if prices else None,
        "avg_rounds": _mean([r["rounds"] for r in runs]),
        "avg_rounds_deals_only": _mean([r["rounds"] for r in deals]),
        "timeout_rate": sum(r["status"] == "timeout" for r in runs) / n if n else None,
        "tag_compliance": _mean([float(x) for x in all_tags]),
        "avg_upstream_global_score": _mean([r["upstream_scores"]["global"] for r in runs]),
        "avg_upstream_buyer_score": _mean([r["upstream_scores"]["buyer"] for r in runs]),
        "avg_upstream_seller_score": _mean([r["upstream_scores"]["seller"] for r in runs]),
        "avg_paper_buyer_score": _mean([r["paper_alg1_scores"]["buyer"] for r in runs]),
    }
