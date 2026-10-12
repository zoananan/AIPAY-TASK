"""按论文附录 Table 14（买方）/ Table 15（卖方）拼装 prompt 的 Agent。

为什么不直接用上游的 BuyerAgent / SellerAgent？
论文对应版本（1a347cf）的上游 prompt 与 Table 14/15 有几处出入：
  1. 上游卖方 prompt 没有 "DEAL AGREEMENT INSTRUCTION / MAKE_DEAL" 段，而 Table 15 有；
  2. 上游第一句会拼成 "You are Buyer, You are a buyer looking ..."（role_description 重复了 "You are"），
     Table 14 是 "You are Buyer, a buyer looking ..."；
  3. 上游把整个 observation（其中又含一遍完整对话历史）塞进 "Current Negotiation State"，
     导致对话历史在 prompt 里出现两次；Table 14/15 中 {current_state} 与 {conversation_history} 是两个独立槽位；
  4. 上游 temperature=0.0 写死在 respond() 里，无法做“10 次独立运行”。
任务书要求“从 Table 14、15 的默认 prompt 出发”，所以模板逐字取自论文表格（prompts/*.txt），
其余部分尽量复用上游代码：
  - 继承上游 BuyerAgent / SellerAgent（initialize()、context、buyer_max_price 等原样复用）；
  - {context}/{current_state}/{conversation_history} 用上游 BaseAgent 的 _format_context /
    _format_state / _format_history 原样格式化；
  - 去掉 <think> 标签的后处理与上游一致。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from exp_config import PROMPT_DIR, ExperimentConfig
from upstream import BaseLLM, BuyerAgent, SellerAgent


def _load_template(name_or_path: str) -> str:
    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        path = PROMPT_DIR / name_or_path
    return path.read_text(encoding="utf-8")


def _block(text: str) -> str:
    """多行内容另起一行，与上游 BaseAgent._build_prompt 的排版一致（标题后换行）。"""
    return "\n" + text.rstrip()


def _insert_persona(prompt: str, persona: str, name: str) -> str:
    """把人物资料插在最后一行 "Now, respond as {name}:" 之前；persona 为空时 prompt 与论文表格完全一致。"""
    if not persona.strip():
        return prompt
    tail = f"Now, respond as {name}:"
    head, sep, rest = prompt.rpartition(tail)
    if not sep:
        return prompt.rstrip() + "\n\n" + persona.strip() + "\n"
    return f"{head}{persona.strip()}\n\n{sep}{rest}"


def _strip_think(response: str) -> str:
    # 与上游 buyer_agent.py / seller_agent.py 相同的后处理
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()


class _PaperPromptMixin:
    """买卖双方共用的部分。self 同时是上游 BaseAgent 的子类，所以能直接调用其格式化方法。"""

    template: str
    persona: str
    sampling: Dict[str, Any]
    last_prompt: str = ""

    def _common_slots(self, conversation_history: List[Dict[str, Any]],
                      current_state: Dict[str, Any]) -> Dict[str, Any]:
        # 上游 observation 里自带一份 conversation_history；它已有独立槽位，这里从 state 中去掉避免重复
        public_state = {k: v for k, v in current_state.items() if k != "conversation_history"}
        return {
            "name": self.name,
            "context": _block(self._format_context()),                      # 上游 BaseAgent._format_context
            "current_state": _block(self._format_state(public_state)),      # 上游 BaseAgent._format_state
            "conversation_history": _block(self._format_history(conversation_history)),  # 上游 _format_history
            "product_info": self.context.get("product_info", {}),
            "available_products": self.context.get("available_products") or "None",
            "environment_info": self.context.get("environment_info", {}),
        }

    def _generate(self, prompt: str) -> str:
        self.last_prompt = prompt                      # 保存下来写进日志，方便核对模型实际看到了什么
        return _strip_think(self.model.generate(prompt, **self.sampling))


class PaperBuyerAgent(_PaperPromptMixin, BuyerAgent):
    """论文 Table 14 买方。对应论文 §3 的买方策略 π^B(· | b_i, v_j, x, h^(t-1))。"""

    def __init__(self, model: BaseLLM, name: str, buyer_max_price: float,
                 template: str = "buyer_table14.txt", persona: str = "",
                 sampling: Optional[Dict[str, Any]] = None):
        super().__init__(model=model, name=name, buyer_max_price=buyer_max_price)
        self.template = _load_template(template)
        self.persona = persona
        self.sampling = sampling or {"temperature": 0.0, "max_tokens": 1024}

    def _preference_guidance(self) -> str:
        profile = self.context.get("user_profile")
        return str(profile) if profile else "None"

    def build_prompt(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        slots = self._common_slots(conversation_history, current_state)
        slots["max_price"] = self.buyer_max_price or self.context.get("max_price", "unknown")  # 买方私有 b_i
        slots["preference_guidance"] = self._preference_guidance()
        return _insert_persona(self.template.format(**slots), self.persona, self.name)

    def respond(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")
        return self._generate(self.build_prompt(conversation_history, current_state))


class PaperSellerAgent(_PaperPromptMixin, SellerAgent):
    """论文 Table 15 卖方。对应论文 §3 的卖方策略 π^S(· | σ_j, v_j, x, h^(t-1))。"""

    def __init__(self, model: BaseLLM, name: str, seller_min_price: float,
                 template: str = "seller_table15.txt", persona: str = "",
                 sampling: Optional[Dict[str, Any]] = None):
        super().__init__(model=model, name=name, seller_min_price=seller_min_price)
        self.template = _load_template(template)
        self.persona = persona
        self.sampling = sampling or {"temperature": 0.0, "max_tokens": 1024}

    def build_prompt(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        slots = self._common_slots(conversation_history, current_state)
        slots["initial_price"] = self.context.get("initial_price", "unknown")
        slots["min_price"] = self.seller_min_price or self.context.get("min_price", "unknown")  # 卖方私有 σ_j
        return _insert_persona(self.template.format(**slots), self.persona, self.name)

    def respond(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")
        return self._generate(self.build_prompt(conversation_history, current_state))


def build_agents(cfg: ExperimentConfig, model: BaseLLM):
    """同一个模型对象分别扮演买方和卖方：两名 Agent 只在 prompt（角色 + 私有保留价）上不同。"""
    sampling = cfg.sampling_kwargs()
    buyer = PaperBuyerAgent(model, cfg.buyer_name, cfg.buyer_max_price,
                            template=cfg.buyer_template, persona=cfg.buyer_persona, sampling=sampling)
    seller = PaperSellerAgent(model, cfg.seller_name, cfg.seller_min_price,
                              template=cfg.seller_template, persona=cfg.seller_persona, sampling=sampling)
    return buyer, seller
