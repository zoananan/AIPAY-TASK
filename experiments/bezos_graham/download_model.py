"""一次性下载 Qwen3-4B-Instruct-2507 到本地（需要联网，只运行一次）。之后所有实验都离线加载。

用法（在仓库根目录）：
    python experiments/bezos_graham/download_model.py
    python experiments/bezos_graham/download_model.py --local-dir D:/models/Qwen3-4B-Instruct-2507

国内网络访问 huggingface.co 困难时，可先设置镜像再运行：
    PowerShell:  $env:HF_ENDPOINT = "https://hf-mirror.com"
    bash/zsh  :  export HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import argparse

from exp_config import DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--local-dir", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.repo_id} -> {args.local_dir}  (约 8 GB)")
    path = snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir)
    print(f"Done: {path}")
    print("如果下载到了别的目录，运行实验时加 --model-path <目录>，或设置环境变量 QWEN_MODEL_PATH。")


if __name__ == "__main__":
    main()
