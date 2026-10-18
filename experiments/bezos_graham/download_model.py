"""一次性下载模型到本地（需要联网，只运行一次）。之后所有实验都离线加载。

报告中的全部实验用的是 GGUF Q8_0 文件（CPU 可跑）：
    python experiments/bezos_graham/download_model.py --gguf --local-dir D:/models
    -> D:/models/Qwen3-4B-Instruct-2507-Q8_0.gguf（约 4.28 GB，来自 unsloth/Qwen3-4B-Instruct-2507-GGUF）
    运行实验时：--backend gguf --model-path D:/models/Qwen3-4B-Instruct-2507-Q8_0.gguf
    校验：PowerShell  Get-FileHash D:/models/Qwen3-4B-Instruct-2507-Q8_0.gguf，
          与 Hugging Face 文件页面显示的 SHA256 对比。

原版权重（safetensors，约 8 GB，transformers 后端；CPU 上很慢）：
    python experiments/bezos_graham/download_model.py
    python experiments/bezos_graham/download_model.py --local-dir D:/models/Qwen3-4B-Instruct-2507

国内网络访问 huggingface.co 困难时，可先设置镜像再运行：
    PowerShell:  $env:HF_ENDPOINT = "https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET = "1"
    bash/zsh  :  export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
"""

from __future__ import annotations

import argparse

from exp_config import DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH

GGUF_REPO_ID = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
GGUF_FILENAME = "Qwen3-4B-Instruct-2507-Q8_0.gguf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", action="store_true",
                        help=f"下载报告所用的 {GGUF_FILENAME}（来自 {GGUF_REPO_ID}）")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    if args.gguf:
        from huggingface_hub import hf_hub_download

        repo_id = args.repo_id or GGUF_REPO_ID
        local_dir = args.local_dir or str(DEFAULT_MODEL_PATH.parent)
        print(f"Downloading {repo_id}/{GGUF_FILENAME} -> {local_dir}  (约 4.28 GB)")
        path = hf_hub_download(repo_id=repo_id, filename=GGUF_FILENAME, local_dir=local_dir)
        print(f"Done: {path}")
        print(f"运行实验时加：--backend gguf --model-path \"{path}\"")
        return

    from huggingface_hub import snapshot_download

    repo_id = args.repo_id or DEFAULT_MODEL_ID
    local_dir = args.local_dir or str(DEFAULT_MODEL_PATH)
    print(f"Downloading {repo_id} -> {local_dir}  (约 8 GB)")
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Done: {path}")
    print("如果下载到了别的目录，运行实验时加 --model-path <目录>，或设置环境变量 QWEN_MODEL_PATH。")


if __name__ == "__main__":
    main()
