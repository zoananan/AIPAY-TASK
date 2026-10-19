# AIPAY-TASK

用 [AgenticPay](https://github.com/SafeRL-Lab/AgenticPay)（arXiv:2602.06008）模拟 **Jeff Bezos（买方）向 Donald Graham（卖方）收购《华盛顿邮报》** 的一对一议价，双方由同一个本地离线模型 **Qwen3-4B-Instruct-2507**（Q8_0 GGUF）扮演。买方秘密上限 350、卖方秘密底价 200（单位：百万美元），成交规则与论文一致（双方报价完全相同才成交）。

*Simulating Jeff Bezos (buyer) negotiating to buy The Washington Post from Donald Graham (seller) with AgenticPay, both sides played offline by Qwen3-4B-Instruct-2507. English report: [`docs/05_report_en.md`](docs/05_report_en.md).*

## 报告 / Reports

- 中文：[`docs/05_实验报告.md`](docs/05_实验报告.md)
- English: [`docs/05_report_en.md`](docs/05_report_en.md)

## 结果 / Results（每组 10 次独立运行，种子 0–9）

| 组 Group | 改动 Change | 成交率 Deals | 平均成交价 Avg. price | 平均轮数 Avg. rounds | 论文 BuyerScore |
|---|---|---|---|---|---|
| A0_paper_default | 论文 prompt（Table 14/15） | 3/10 | 330.0 | 15.0 | 13.5 |
| A0_personas | + 双方公开资料（第 1 部分） | 1/10 | 315.0 | 18.4 | 3.3 |
| A1 | + 买方策略（第 2 部分 v5） | 8/10 | 289.7 | 7.2 | 51.6 |
| A2 | + 买方策略 + 议价筹码 | 6/10 | 278.3 | 10.4 | 40.5 |
| A1-nod（消融） | A1 去掉第 8 轮截止条款 | 10/10 | 286.3 | 3.8 | 66.5 |

平均成交价只计成交的运行；平均轮数计全部 10 次（超时按 20 轮）。原始结果见 [`experiments/bezos_graham/results/`](experiments/bezos_graham/results/)。

## 文档 / Docs

| 文件 | 内容 |
|---|---|
| [`docs/01_论文与代码对照.md`](docs/01_论文与代码对照.md) | 论文各节对应的源码位置；论文版代码与论文不一致之处 |
| [`docs/02_第一步代码讲解.md`](docs/02_第一步代码讲解.md) | 本仓库代码逐文件讲解、运行步骤 |
| [`docs/03_任务第1部分_人物资料与来源.md`](docs/03_任务第1部分_人物资料与来源.md) | 第 1 部分：资料抓取流程、来源、筛选规则、加入的事实 |
| [`docs/04_任务第2部分_买方prompt设计.md`](docs/04_任务第2部分_买方prompt设计.md) | 第 2 部分：prompt 设计、迭代过程、结果与机制分析 |
| [`docs/05_实验报告.md`](docs/05_实验报告.md) / [`docs/05_report_en.md`](docs/05_report_en.md) | 完整报告（中文 / English），含真实场景下的问题讨论 |
| [`docs/06_补充研究思考.md`](docs/06_补充研究思考.md) / [`docs/06_supplementary_notes_en.md`](docs/06_supplementary_notes_en.md) | 补充研究思考（中文 / English）：中间价与纳什议价、多方市场剩余、因果解释与样本量、算力与可扩展性、博弈论分析框架 |
| [`docs/word/`](docs/word/) | 05、06 两份报告的 Word 版（由 Markdown 用 pandoc 转换） |

## 复现 / Reproduce（Windows PowerShell，CPU 即可）

```powershell
git clone -b claude/eager-hypatia-8txpol https://github.com/zoananan/AIPAY-TASK.git
cd AIPAY-TASK
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 不需要模型：单元测试 + mock 流程
python -m unittest discover -s experiments/bezos_graham/tests -v
python experiments/bezos_graham/run_episode.py --backend mock --personas --show-prompt

# 下载报告所用的模型文件（只需联网一次；国内可先设置 $env:HF_ENDPOINT="https://hf-mirror.com"）
python experiments/bezos_graham/download_model.py --gguf --local-dir D:\models

# 第 1 部分：用已缓存的快照与模型输出离线重建人物资料
python experiments/bezos_graham/persona_pipeline.py extract --llm none
python experiments/bezos_graham/persona_pipeline.py build

# 第 2 部分：各组 10 次运行（示例为 A1；其余组的命令见报告附录 B）
$M = "D:\models\Qwen3-4B-Instruct-2507-Q8_0.gguf"
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A1_disciplined --quiet --backend gguf --model-path $M --max-new-tokens 300 --personas --buyer-template buyer_A1_disciplined.txt
python experiments/bezos_graham/analyze_closing.py experiments/bezos_graham/outputs/<结果文件夹>
```

## 目录 / Layout

```
third_party/AgenticPay/          官方源码原样拷贝（论文版本 1a347cf，未修改，见 VENDORED.md）
experiments/bezos_graham/        实验代码
  prompts/                       论文 prompt（原文）与买方变体 A1 / A2 / A1-nod
  personas/                      人物资料、来源表、候选事实审计表、快照与模型输出缓存
  results/                       各组 summary.json、runs.csv、代表性对话记录
  tests/                         单元测试
docs/                            文档与报告
```
