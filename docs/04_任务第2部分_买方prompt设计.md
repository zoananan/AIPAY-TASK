# 任务第 2 部分：只改买方 prompt，提升 Bezos 的议价能力

> Task: *change only the buyer's prompt to improve Bezos's bargaining power, meaning a lower final price. Try anything that makes sense, like 1-2 variants plus the original, and explain how you design your new prompt. For each version, run 10 independent runs of negotiation and report the average final price, how often a deal was made, and the average number of rounds per run.*

## 1. 控制变量

三组实验之间**只有买方模板不同**，其余完全相同：

| 项目 | 三组共同设置 |
|---|---|
| 环境与成交规则 | 论文版 Task1 环境，`price_tolerance = 0`（报价完全相同才成交），最多 20 轮 |
| 卖方 prompt | Table 15 原文 + Graham 人物资料（`--personas`），三组一字不差 |
| 买方 prompt 的基础部分 | Table 14 原文 + Bezos 人物资料；变体只在此基础上**增加**一段策略 |
| 模型与采样 | Qwen3-4B-Instruct-2507 Q8_0 GGUF，temperature 0.7 / top_p 0.8 / top_k 20，seed 0–9，`--max-new-tokens 300` |

单元测试 `TestBuyerVariants` 会检查：两个变体都完整保留 Table 14 的每一行，且卖方 prompt 与 A0 完全相同。

| 组别 | 买方模板 | 设计 |
|---|---|---|
| **A0（original）** | `prompts/buyer_table14.txt` | 论文原文 |
| **A1（策略纪律）** | `prompts/buyer_A1_disciplined.txt` | A0 + “谈判策略”段 |
| **A2（策略纪律 + 议价筹码）** | `prompts/buyer_A2_leverage.txt` | A1 + “议价筹码”段 |

A2 = A1 + 一段新内容，所以 **A1 − A0** 衡量“谈判纪律”的效果，**A2 − A1** 衡量“议价筹码”的额外效果。

## 2. 设计依据：基线里观察到的买方弱点

论文原始设定下的 10 次运行（不含人物资料）暴露了三个问题，每条新规则都对应其中一个：

| 观察到的问题 | 证据 | 对应的新规则 |
|---|---|---|
| ① 开价太高、让步太快 | 买方首轮出 280–300，两三轮就让到 335–350（上限 350） | 低锚定（约上限的 60%）、每次最多加 10、加价幅度递减 |
| ② 单方面让步 | 卖方不动时，买方仍继续加价 | 互惠：卖方不降，买方不加 |
| ③ 不懂成交机制，卡在“差一点” | 7 次超时中 6 次只差 2.5–5；双方都写了 MAKE_DEAL，但报价不同，论文版环境不认 | 明确告诉买方：只有报价完全相同才成交，MAKE_DEAL 本身不算；收尾时提出一个具体数字，或原样复述卖方报价来接受 |
| ④ 主动把高价说成“公平” | 买方多次说接近卖方报价的价格“fair and balanced” | 不因为价格接近卖方报价就称其公平；不超过卖方最新报价 |

A2 再加入**议价筹码**，全部基于第 1 部分的公开事实，没有编造：

- 报纸部门亏损、收入和发行量多年下滑（2012 年 10-K）→ 价格应反映需要多年投入的业务；
- Graham 公开表示希望报纸“成功而不只是生存”→ 把长期投入和资源作为自己带来的价值，以此替代高价；
- 表明可以放弃交易（walk-away），在卖方停止让步时冷静说明。

## 3. 设计上的取舍

- **只给原则，不写死数字**：买方只知道自己的上限，所以锚定用“上限的 60%”这种相对说法，不需要另外暴露信息。
- **不改成交规则**：僵局问题通过让买方理解规则来解决，而不是放宽环境。
- **保留 Table 14 原文**：包括 MAKE_DEAL 的说明；新段落只是补充“MAKE_DEAL 不能单独促成交易”。
- **可能的副作用**：互惠和慢让步可能让一些运行谈满 20 轮，导致成交率下降。报告里要同时看平均成交价和成交率：只降价不成交并不是更好的结果。

### 3.1 试运行后的修订（v1 → v2）

A1 第一版（v1，提交 `cb3e808`）在正式实验的第 1 次运行（seed 0）中暴露了一个设计缺陷：卖方从第 4 轮起停在 290，买方严格执行 v1 的互惠规则（“卖方不降，买方不加”），从第 5 轮起停在 285，双方死锁。v1 的收尾规则虽然允许买方接受卖方报价，但模型优先执行了互惠规则。这一次运行在第 10 轮左右被手动中止，没有计入结果，只作为试运行记录。

v2 只改了策略段中的三处，其余内容不变：

1. 互惠规则只在差距大于 10 时生效；
2. 收尾规则改为两步：差距 ≤ 10 时，先提出一个中间价；卖方下一条消息没有写出这个数字，且卖方报价不超过买方上限，就原样复述卖方报价并写 MAKE_DEAL；
3. 告诉买方谈满 20 轮不成交是最差结果（这是环境的真实规则，不涉及对方的秘密信息）。

A2 包含与 A1 完全相同的策略段，所以同步更新。A1、A2 的 10 次正式运行都使用 v2。v1 的试运行说明：“压价”和“成交”之间存在取舍，过于刚性的互惠规则会在卖方不让步时导致死锁。

## 4. 运行命令

三组都在同一台电脑、同样的参数下运行：

```powershell
# A0：论文原文 + 人物资料
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A0_personas --quiet --backend gguf --model-path D:\models\Qwen3-4B-Instruct-2507-Q8_0.gguf --max-new-tokens 300 --personas

# A1：+ 谈判策略
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A1_disciplined --quiet --backend gguf --model-path D:\models\Qwen3-4B-Instruct-2507-Q8_0.gguf --max-new-tokens 300 --personas --buyer-template buyer_A1_disciplined.txt

# A2：+ 谈判策略 + 议价筹码
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A2_leverage --quiet --backend gguf --model-path D:\models\Qwen3-4B-Instruct-2507-Q8_0.gguf --max-new-tokens 300 --personas --buyer-template buyer_A2_leverage.txt
```

每组结束后，`outputs/<tag>_gguf_<时间>/summary.json` 里就是任务要求的三个指标：`avg_final_price`、`deal_rate`、`avg_rounds`。

### 可选：`--cache-gb 2`（未验证能提速）

买卖双方轮流使用同一个模型，两人的 prompt 开头不同，llama.cpp 默认只能复用"上一次"的前缀，所以每轮都要把整段对话从头读一遍。`--cache-gb 2` 会打开 2 GB 的 prompt 缓存，让两人各自读过的部分被保留。

我在开发环境里用一个小模型做了对比：开不开缓存，模型生成的内容**完全相同**（8 句话逐字一致），但那次测试的 prompt 只有约 1,500 个 token，**没有测出提速**。对话越长，缓存理论上越有用，但效果没有实测证明。如果要用，三组必须都加，或者都不加。
