"""运行一次完整谈判（一个 episode）并把全部过程保存下来。

主循环逐行对应上游示例
third_party/AgenticPay/agenticpay/examples/single_buyer_product_seller/Task1_basic_price_negotiation.py
第 167–225 行（env.reset → while: buyer.respond → seller.respond → env.step）。
环境本身（报价解析、成交判断、评分）完全使用上游 Task1BasicPriceNegotiation，没有改动。
"""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

from exp_config import ExperimentConfig
from metrics import audit_message, paper_algorithm1_scores
from paper_agents import build_agents
from upstream import UPSTREAM_COMMIT, BaseLLM, make


def create_env(cfg: ExperimentConfig, buyer, seller):
    """[上游] 通过注册表创建 1 买 1 卖 1 商品的基础议价环境（论文 §4.1–4.2，Table 11 第一组）。"""
    return make(
        "Task1_basic_price_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=cfg.env_max_rounds(),
        initial_seller_price=cfg.initial_asking_price,   # 进入卖方 context → Table 15 的 {initial_price}
        buyer_max_price=cfg.buyer_max_price,             # 只进入买方 context（私有 b_i）
        seller_min_price=cfg.seller_min_price,           # 只进入卖方 context（私有 σ_j）
        environment_info=cfg.environment_info,           # 双方共享的市场信息 x
        price_tolerance=cfg.price_tolerance,
        # 论文 §5.1 的评分参数（GlobalScore / BuyerScore / SellerScore）
        gamma=cfg.gamma,
        deal_score_weight=cfg.deal_weight_D, quality_score_weight=cfg.quality_weight_W,
        efficiency_score_weight=cfg.efficiency_weight_E, failure_penalty_weight=cfg.failure_penalty_F,
        buyer_deal_weight=cfg.deal_weight_D, buyer_utility_weight=cfg.quality_weight_W,
        buyer_efficiency_weight=cfg.efficiency_weight_E, buyer_failure_penalty_weight=cfg.failure_penalty_F,
        seller_deal_weight=cfg.deal_weight_D, seller_utility_weight=cfg.quality_weight_W,
        seller_efficiency_weight=cfg.efficiency_weight_E, seller_failure_penalty_weight=cfg.failure_penalty_F,
    )


def _turn_record(env, model: BaseLLM, role: str, round_idx: int, text: str, reservation: float) -> Dict[str, Any]:
    return {
        "round": round_idx + 1,                       # 1 起算，便于阅读
        "role": role,
        "text": text,
        "env_parsed_price": env._extract_price(text),  # 上游解析器 Π 对这条消息的解析结果
        "audit": audit_message(text, role, reservation),
        "gen": dict(getattr(model, "last_gen_info", {}) or {}),
    }


def run_episode(cfg: ExperimentConfig, model: BaseLLM, run_idx: int, out_dir: Path) -> Dict[str, Any]:
    seed = cfg.base_seed + run_idx
    if hasattr(model, "reseed"):
        model.reseed(seed)                 # 每次运行独立的随机种子 → 10 次运行互相独立且各自可复现

    buyer, seller = build_agents(cfg, model)
    env = create_env(cfg, buyer, seller)

    # [上游] reset：清空记忆、轮次归零、把私有保留价分别注入买卖双方的 context
    observation, info = env.reset(
        user_requirement=cfg.user_requirement,
        product_info=cfg.product_info,
        user_profile=None,                 # Table 14 的 USER PREFERENCES；基线不设置
    )

    turns: List[Dict[str, Any]] = []
    first_prompts: Dict[str, str] = {}
    env_stdout = io.StringIO()
    t_start = time.time()
    done = False
    if cfg.verbose:
        print(f"\n######## run {run_idx} (seed={seed}) ########")

    while not done:
        round_idx = observation["current_round"]

        # 1) 买方先说话：π^B(· | b_i, v_j, x, h^(t-1))
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation,
        )
        turns.append(_turn_record(env, model, "buyer", round_idx, buyer_action, cfg.buyer_max_price))

        # 2) 卖方能看到买方本轮刚说的话（与上游示例相同的做法）
        seller_history = observation["conversation_history"].copy()
        seller_history.append({"role": "buyer", "content": buyer_action, "round": round_idx})
        seller_action = seller.respond(conversation_history=seller_history, current_state=observation)
        turns.append(_turn_record(env, model, "seller", round_idx, seller_action, cfg.seller_min_price))

        if not first_prompts:
            first_prompts = {"buyer": buyer.last_prompt, "seller": seller.last_prompt}

        # 3) [上游] 环境解析报价（解析器 Π）→ 判断成交/超时 → 终局时计算评分
        with redirect_stdout(env_stdout):                 # 上游会 print 奖励明细，收进日志而不刷屏
            observation, reward, terminated, truncated, info = env.step(
                buyer_action=buyer_action, seller_action=seller_action
            )
        done = terminated or truncated
        if cfg.verbose:
            print(env.render(mode="text"))               # [上游] 每轮的对话与价格摘要

    elapsed = time.time() - t_start
    agreed = info["status"] == "agreed"
    price = info["agreed_price"]
    rounds = info["round"]                                # 已完成的轮数（成交轮 / 超时时为最大轮数）

    record: Dict[str, Any] = {
        "run_idx": run_idx,
        "seed": seed,
        "status": info["status"],
        "deal": agreed,
        "agreed_price": price,
        # 上游的成交判断不检查价格是否落在 [200, 350]；论文 §4.1 要求成交价必须在议价区间内
        "deal_in_zone": bool(agreed and price is not None
                             and cfg.seller_min_price <= price <= cfg.buyer_max_price),
        "rounds": rounds,
        "final_buyer_offer": info["buyer_price"],
        "final_seller_offer": info["seller_price"],
        "upstream_scores": {
            "global": info.get("global_score"),
            "buyer": info.get("buyer_score"),
            "seller": info.get("seller_score"),
        },
        "paper_alg1_scores": paper_algorithm1_scores(
            price, rounds, agreed, cfg.buyer_max_price, cfg.seller_min_price, cfg.max_rounds,
            cfg.deal_weight_D, cfg.quality_weight_W, cfg.efficiency_weight_E, cfg.failure_penalty_F, cfg.gamma,
        ),
        "elapsed_s": round(elapsed, 1),
        "n_generations": len(turns),
        "turns": turns,
        "first_prompts": first_prompts,
        "upstream_env_log": env_stdout.getvalue(),
        "model": repr(model),
        "agenticpay_commit": UPSTREAM_COMMIT,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"run_{run_idx:02d}"
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}_transcript.txt").write_text(format_transcript(record, cfg), encoding="utf-8")
    env.close()
    return record


def format_transcript(record: Dict[str, Any], cfg: ExperimentConfig) -> str:
    lines = [
        f"Run {record['run_idx']}  seed={record['seed']}  status={record['status']}  "
        f"agreed_price={record['agreed_price']}  rounds={record['rounds']}",
        f"buyer max={cfg.buyer_max_price:g}  seller min={cfg.seller_min_price:g}  "
        f"initial ask={cfg.initial_asking_price:g}  (USD millions)",
        "=" * 80,
    ]
    for t in record["turns"]:
        name = cfg.buyer_name if t["role"] == "buyer" else cfg.seller_name
        a = t["audit"]
        flags = f"parsed={t['env_parsed_price']}  tag_ok={a['tag_ok']}  MAKE_DEAL={a['make_deal']}"
        if a["possible_leak"]:
            flags += "  POSSIBLE_RESERVATION_LEAK"
        lines += [f"[Round {t['round']}] {t['role'].upper()} ({name})   <{flags}>", t["text"], ""]
    return "\n".join(lines)
