# Experiment results / 实验结果

All runs: Qwen3-4B-Instruct-2507, Q8_0 GGUF (`unsloth/Qwen3-4B-Instruct-2507-GGUF`, file `Qwen3-4B-Instruct-2507-Q8_0.gguf`), llama-cpp-python on CPU, temperature 0.7 / top_p 0.8 / top_k 20, seeds 0–9, `--max-new-tokens 300`, AgenticPay commit 1a347cf, `price_tolerance = 0`, 20 rounds max. Prices in USD millions; buyer max 350, seller min 200, initial ask 400.

所有运行的设置同上。完整的 `run_XX.json` / 记录保存在运行者本机的 `outputs/` 中（体积较大，未全部入库）；这里收录每组的汇总、逐局表和代表性记录。

| Folder | Buyer prompt | Personas | Files |
|---|---|---|---|
| `A0_paper_default_q8/` | `buyer_table14.txt` (paper) | no | summary, runs |
| `A0_personas/` | `buyer_table14.txt` (paper) | yes | summary, runs, transcripts of runs 3, 4, 7, 8 |
| `A1_disciplined/` | `buyer_A1_disciplined.txt` (v5) | yes | summary, runs, runs 0–2 (json + transcript) |
| `A2_leverage/` | `buyer_A2_leverage.txt` (v5) | yes | summary, runs, transcript of run 7 (price overflow) |
| `A1nod/` | `buyer_A1_nodeadline.txt` (v5, no deadline clause) | yes | summary, runs |
| `A1_A2_closing_analysis_output.txt`, `A1nod_closing_analysis_output.txt` | — | — | `analyze_closing.py` terminal output |
| `pilots/A1v3_pilot/` | A1 v3 (git `09ce6df`) | yes | aborted pilot, seed 0: buyer sent only price tags (not counted) |
| `pilots/A1v4_smoke/` | A1 v4 (git `07dc069`) | yes | 6-round smoke test, seed 0: buyer copied its own message (not counted) |

The seller prompt is identical in all persona groups (Table 15 + Graham facts). Each run's buyer and seller prompts are stored inside `run_XX.json` (`first_prompts`).
