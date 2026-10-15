"""把 Qwen3-4B-Instruct-2507 接入 AgenticPay 的统一 LLM 接口（上游 models/base_llm.py::BaseLLM）。

上游 Agent 只调用一个方法：model.generate(prompt, temperature=..., max_tokens=...) -> str。
所以只要实现这个接口，模型就能“即插即用”地扮演买方和卖方（论文 §5.1 “drop-in policy”）。

提供三种后端：
- QwenHFLLM   ：HuggingFace transformers，本地离线加载，Windows / macOS / Linux 通用（推荐）。
- QwenVLLMLLM ：vLLM，仅 Linux + NVIDIA GPU，速度快。
- MockNegotiatorLLM：不加载任何模型的“脚本谈判者”，用于在没有 GPU 时先跑通整条流程。

与上游 models/vllm_lm.py::VLLMLLM 的关键区别：
上游把整段 prompt 当作“裸文本”直接续写（llm.generate([prompt])），没有套用聊天模板；
而 Qwen3-*-Instruct 是对话模型，必须用 chat template 包装成
<|im_start|>user ... <|im_end|><|im_start|>assistant，否则输出质量明显下降。
上游的 API 后端（custom_llm.py / openai_llm.py，论文里专有模型走的路径）是把整段 prompt
作为一条 {"role": "user"} 消息发送的，我们与之保持一致。
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from exp_config import ExperimentConfig
from upstream import BaseLLM


def _messages(prompt: str) -> List[Dict[str, str]]:
    # 与上游 API 后端一致：整段 prompt 作为一条 user 消息，不加 system 消息
    return [{"role": "user", "content": prompt}]


class QwenHFLLM(BaseLLM):
    """transformers 后端（本地目录加载，可完全离线）。"""

    def __init__(self, model_path: str, device: str = "auto", offline: bool = True, dtype: str = "auto"):
        self.model_path = model_path
        self.device_pref = device
        self.dtype_pref = dtype
        self.offline = offline
        self._tok = None
        self._model = None
        self.last_gen_info: Dict[str, Any] = {}

    # ---- 懒加载：第一次 generate 时才把模型读进显存 ----
    def _load(self) -> None:
        if self._model is not None:
            return
        if self.offline:
            # 必须在 import transformers 之前设置，才能保证全程不联网
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"模型目录不存在：{self.model_path}\n请先运行：python experiments/bezos_graham/download_model.py"
            )

        if self.device_pref == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device_pref

        # transformers 4.56 起把 torch_dtype 改名为 dtype
        # dtype："auto" = 模型自带的 bfloat16。很多 CPU 没有 bf16 硬件指令，跑 bf16 反而很慢，
        # 这时用 --dtype float32 通常快好几倍（代价是内存翻倍：4B 模型约 16 GB）。
        dtype = "auto" if self.dtype_pref == "auto" else getattr(torch, self.dtype_pref)
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}

        print(f"[model] loading {self.model_path} on {device} (transformers {transformers.__version__}) ...")
        t0 = time.time()
        self._tok = AutoTokenizer.from_pretrained(self.model_path, local_files_only=self.offline)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=self.offline, **dtype_kw
        ).to(device)
        self._model.eval()
        self._device = device
        print(f"[model] loaded in {time.time() - t0:.1f}s, dtype={self._model.dtype}")

    def reseed(self, seed: int) -> None:
        """每次独立运行开始时调用一次，使这一次运行可复现。"""
        from transformers import set_seed
        set_seed(seed)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        self._load()
        import torch

        # 套用 Qwen 的聊天模板（上游 VLLMLLM 缺少这一步）
        text = self._tok.apply_chat_template(_messages(prompt), tokenize=False, add_generation_prompt=True)
        inputs = self._tok([text], return_tensors="pt").to(self._device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_tokens or 1024}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            if top_k:
                gen_kwargs["top_k"] = top_k
        else:
            # 贪心解码（论文 §5.1 的 temperature 0）；把模型自带的采样默认值清掉，避免警告
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        t0 = time.time()
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        self.last_gen_info = {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "new_tokens": int(new_tokens.shape[0]),
            "seconds": round(time.time() - t0, 2),
        }
        return self._tok.decode(new_tokens, skip_special_tokens=True).strip()

    def __repr__(self) -> str:
        return f"QwenHFLLM(model={self.model_path})"


class QwenGGUFLLM(BaseLLM):
    """llama.cpp 后端（llama-cpp-python），加载量化后的 GGUF 单文件，适合没有 NVIDIA 显卡、内存 16 GB 的电脑。

    - 模型：同一个 Qwen3-4B-Instruct-2507，转换为 GGUF 并量化（如 Q8_0，约 4.3 GB）；
    - 聊天模板：GGUF 文件里自带 Qwen 的 chat template，create_chat_completion 会自动套用，
      与其他后端一样把整段 prompt 作为一条 user 消息；
    - 随机种子：与 vLLM 后端相同的做法，每次调用用 (运行 seed, 调用序号) 组合，整次运行可复现。
    """

    def __init__(self, model_path: str, n_ctx: int = 16384, n_threads: Optional[int] = None,
                 cache_gb: float = 0.0):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"GGUF 模型文件不存在：{model_path}\n"
                "--model-path 需要指向 .gguf 文件本身，例如 D:\\models\\Qwen3-4B-Instruct-2507-Q8_0.gguf"
            )
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "需要先安装 llama-cpp-python（CPU 版）：\n"
                "pip install llama-cpp-python==0.3.36 --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary llama-cpp-python"
            ) from e

        self.model_path = model_path
        # n_ctx：上下文长度。20 轮对话约 1 万 token，16k 足够；KV 缓存约占 2 GB 内存
        # n_threads：默认用全部物理核心
        threads = n_threads or max(1, (os.cpu_count() or 2) // 2)
        print(f"[model] loading GGUF {model_path} (n_ctx={n_ctx}, threads={threads}) ...")
        t0 = time.time()
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads, verbose=False)
        print(f"[model] loaded in {time.time() - t0:.1f}s")
        if cache_gb > 0:
            # 买卖双方轮流使用同一个模型，prompt 开头不同，llama.cpp 默认只能复用"上一次"的前缀，
            # 所以每次都要从头读整段对话。RAM 缓存会按前缀保存两人各自读过的状态，下次只读新增部分。
            # 只影响速度，不改变 prompt、规则和采样参数；三组实验应使用相同设置。
            from llama_cpp import LlamaRAMCache
            self._llm.set_cache(LlamaRAMCache(capacity_bytes=int(cache_gb * (1 << 30))))
            print(f"[model] prompt cache enabled ({cache_gb} GB)")
        self._seed = 0
        self._calls = 0
        self.last_gen_info: Dict[str, Any] = {}

    def reseed(self, seed: int) -> None:
        self._seed = seed
        self._calls = 0

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None,
                 top_p: float = 1.0, top_k: Optional[int] = None, **kwargs: Any) -> str:
        self._calls += 1
        sampled = bool(temperature and temperature > 0)
        t0 = time.time()
        out = self._llm.create_chat_completion(
            messages=_messages(prompt),
            temperature=temperature if sampled else 0.0,
            top_p=top_p if sampled else 1.0,
            top_k=(top_k or 0) if sampled else 1,     # top_k=1 即贪心解码
            min_p=0.0,                                # Qwen 模型卡推荐 MinP=0（llama.cpp 默认 0.05）
            max_tokens=max_tokens or 1024,
            seed=self._seed * 100003 + self._calls,
        )
        usage = out.get("usage", {})
        self.last_gen_info = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "new_tokens": usage.get("completion_tokens"),
            "seconds": round(time.time() - t0, 2),
        }
        return (out["choices"][0]["message"]["content"] or "").strip()

    def __repr__(self) -> str:
        return f"QwenGGUFLLM(model={self.model_path})"


class QwenVLLMLLM(BaseLLM):
    """vLLM 后端（可选，仅 Linux + NVIDIA GPU）。用 llm.chat() 自动套用聊天模板。"""

    def __init__(self, model_path: str, offline: bool = True, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 32768):
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from vllm import LLM

        self.model_path = model_path
        # 模型原生支持 262k 上下文；20 轮对话远用不到，限制到 32k 以节省显存（模型卡的建议）
        self._llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=max_model_len)
        self._seed = 0
        self._calls = 0
        self.last_gen_info: Dict[str, Any] = {}

    def reseed(self, seed: int) -> None:
        self._seed = seed
        self._calls = 0

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None,
                 top_p: float = 1.0, top_k: Optional[int] = None, **kwargs: Any) -> str:
        from vllm import SamplingParams

        # vLLM 的 seed 是“每个请求”的；用 (运行 seed, 调用序号) 组合，保证同一次运行内各轮不同、整次运行可复现
        self._calls += 1
        params = SamplingParams(
            temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
            top_k=(top_k or -1) if temperature > 0 else -1,
            max_tokens=max_tokens or 1024, seed=self._seed * 100003 + self._calls,
        )
        t0 = time.time()
        out = self._llm.chat(_messages(prompt), params, use_tqdm=False)[0]
        self.last_gen_info = {
            "prompt_tokens": len(out.prompt_token_ids),
            "new_tokens": len(out.outputs[0].token_ids),
            "seconds": round(time.time() - t0, 2),
        }
        return out.outputs[0].text.strip()

    def __repr__(self) -> str:
        return f"QwenVLLMLLM(model={self.model_path})"


class MockNegotiatorLLM(BaseLLM):
    """不需要模型的“脚本谈判者”，只用于检查整条流程（prompt 拼装、价格解析、成交判断、结果保存）。

    它只读取自己 prompt 里能看到的信息（自己的保留价 + 公开对话），按随机步长让价，
    对方报价进入自己可接受范围时用相同价格 + MAKE_DEAL 接受。结果没有任何研究意义。
    """

    _TAG = re.compile(r"###\s*(BUYER|SELLER)_PRICE\s*\(\$([\d,]+\.?\d*)\)\s*###", re.I)

    def __init__(self) -> None:
        self.rng = random.Random(0)
        self.last_gen_info: Dict[str, Any] = {}

    def reseed(self, seed: int) -> None:
        self.rng = random.Random(seed)

    @staticmethod
    def _num(pattern: str, text: str) -> Optional[float]:
        m = re.search(pattern, text)
        return float(m.group(1).replace(",", "")) if m else None

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None,
                 **kwargs: Any) -> str:
        is_buyer = "### BUYER_PRICE($X) ###" in prompt
        history = prompt.split("Conversation History:", 1)[-1].split("Please respond naturally", 1)[0]
        offers = [(r.upper(), float(p.replace(",", ""))) for r, p in self._TAG.findall(history)]
        mine = [p for r, p in offers if r == ("BUYER" if is_buyer else "SELLER")]
        theirs = [p for r, p in offers if r != ("BUYER" if is_buyer else "SELLER")]
        step = self.rng.choice([10, 15, 20, 25])

        if is_buyer:
            limit = self._num(r"Your top price is \$([\d,]+(?:\.\d+)?)", prompt) or 0.0
            plan = min(limit, (mine[-1] + step) if mine else self.rng.choice([150, 160, 170, 180]))
            if theirs and theirs[-1] <= plan:
                return f"Fair enough, that works for me. ### BUYER_PRICE(${theirs[-1]:g}) ### MAKE_DEAL"
            return f"I am interested, but the price must be realistic. I can offer ### BUYER_PRICE(${plan:g}) ###."
        floor = self._num(r"minimum acceptable price \(confidential\) is \$([\d,]+(?:\.\d+)?)", prompt) or 0.0
        ask = self._num(r"initial asking price is \$([\d,]+(?:\.\d+)?)", prompt) or floor * 2
        plan = max(floor, (mine[-1] - step) if mine else ask)
        if theirs and theirs[-1] >= plan:
            return f"I accept your offer. ### SELLER_PRICE(${theirs[-1]:g}) ### MAKE_DEAL"
        return f"This is a unique asset. How about ### SELLER_PRICE(${plan:g}) ###?"

    def __repr__(self) -> str:
        return "MockNegotiatorLLM()"


def build_model(cfg: ExperimentConfig) -> BaseLLM:
    """按配置创建模型。买卖双方共享同一个模型对象（任务要求：同一个 LLM 扮演双方）。"""
    if cfg.backend == "hf":
        return QwenHFLLM(cfg.model_path, device=cfg.device, offline=cfg.offline, dtype=cfg.dtype)
    if cfg.backend == "gguf":
        return QwenGGUFLLM(cfg.model_path, cache_gb=cfg.cache_gb)
    if cfg.backend == "vllm":
        return QwenVLLMLLM(cfg.model_path, offline=cfg.offline)
    if cfg.backend == "mock":
        return MockNegotiatorLLM()
    raise ValueError(f"unknown backend: {cfg.backend}")
