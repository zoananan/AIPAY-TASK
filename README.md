# AIPAY-TASK

用 [AgenticPay](https://github.com/SafeRL-Lab/AgenticPay)（arXiv:2602.06008）模拟 **Jeff Bezos（买方）向 Donald Graham（卖方）收购《华盛顿邮报》** 的一对一议价，双方由同一个本地离线模型 **Qwen3-4B-Instruct-2507** 扮演。

当前进度：**第一步**——论文 Table 14/15 默认 prompt，买方秘密上限 350、卖方秘密底价 200（单位：百万美元）。

## 文档

- [`docs/01_论文与代码对照.md`](docs/01_论文与代码对照.md)：论文每一节对应源码的哪个文件、哪几行；论文版代码与论文不一致之处；对已有分析文档的核查
- [`docs/02_第一步代码讲解.md`](docs/02_第一步代码讲解.md)：本仓库代码逐文件讲解、运行步骤、需要确认的假设

## 快速开始

```bash
# 方式一：git clone（需要能访问 GitHub）
git clone -b claude/eager-hypatia-8txpol https://github.com/zoananan/AIPAY-TASK.git
# 方式二：GitHub 网页切到该分支 → Code → Download ZIP，解压（官方代码已包含在 third_party/ 中）
cd AIPAY-TASK
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt                         # 先按 pytorch.org 装好匹配 CUDA 的 torch

python -m unittest discover -s experiments/bezos_graham/tests -v          # 不需要模型
python experiments/bezos_graham/run_episode.py --backend mock --show-prompt   # 不需要模型，跑通流程

python experiments/bezos_graham/download_model.py                         # 只需联网一次（约 8 GB）
python experiments/bezos_graham/run_episode.py --show-prompt              # Qwen3-4B 离线跑 1 次
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A0_paper_default --quiet
```

VS Code 用户可直接使用「运行和调试」面板中的 5 个预设配置（`.vscode/launch.json`）。

## 目录

```
third_party/AgenticPay/     官方源码原样拷贝（论文版本 1a347cf，未修改，见 VENDORED.md）
experiments/bezos_graham/   本任务的实验代码、prompt 模板、测试
docs/                       说明文档
```
