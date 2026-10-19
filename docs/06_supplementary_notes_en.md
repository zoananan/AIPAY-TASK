# Supplementary Research Notes: Scoring, Causal Claims, Compute and Multi-Party Markets

> This document supplements the main report (`docs/05_report_en.md` / `docs/05_实验报告.md`). It collects reflections written after the experiments were finished and does not change any conclusion of the main report. A Chinese version is `docs/06_补充研究思考.md`.
>
> Each section has three parts: **does the paper cover it** (with quotes and code locations), **our analysis**, and **possible extensions**. "The paper" is arXiv 2602.06008v1; "the code" is the paper-era commit `1a347cf` (`third_party/AgenticPay/` in this repository).

## 0. Overview

| Question | Does the paper cover it? | Main point of this document |
|---|---|---|
| 1. Is the midpoint the "optimal" outcome in 1v1? | It only states that GlobalScore peaks when surplus is split equally, without a theoretical justification | The midpoint is a design choice of the metric. It coincides with the symmetric Nash bargaining solution, but other bargaining models or protocol details give different predictions |
| 2. How is surplus measured in multi-party markets? | Multi-party tasks reuse the 1v1 formula and score only the pair that traded | The paper-era code ends the episode at the first deal and does not measure total market surplus or allocative efficiency. It can score an efficient competitive outcome low and an inefficient match at full marks |
| 3. Can A1's advantage over A0 be attributed to the prompt? | Not discussed. Each task is run once, with no repeated runs or significance tests | In our controlled setting, yes: this prompt change caused the difference. It does not generalise to "this particular rule works" or "it works for other models and counterparts" |
| 4. CPU cannot replace GPU; extensibility | The paper uses 4 A800 GPUs and stresses scalability, but does not discuss limited compute | The 50 reported runs took about 36 hours on CPU, which limited sample size, model size and generation length. The code already supports GPU backends; extending it to multi-party tasks needs a few more modules |
| 5. Multi-buyer/multi-seller analysis should move to game-theoretic models | Classical game theory is cited in related work but not used in the analysis; the explanation of multi-party results ("market liquidity") is not tested | Use competitive-equilibrium prices, auction theory, allocative efficiency and stable matching as benchmarks, and design experiments that can test the liquidity explanation |

---

## 1. The "Optimal" Outcome in 1v1: the Midpoint Is a Design Choice, Not a Theorem

### 1.1 Does the paper cover it?

Section 4.4 (Score Design) of the paper states:

> "GlobalScore uses a symmetric quality term Q = 4·rb·rs that peaks when surplus is split equally, while BuyerScore and SellerScore reward each party's individual surplus."

The paper explains that Q is highest at an equal split, but not why an equal split is a good outcome, and it does not relate this to bargaining solutions from game theory. The Nash bargaining solution and Rubinstein's model are not derived or discussed in the main text; the related literature is only cited in Section 2.

### 1.2 Our analysis

1. **Q is the normalised Nash product.** The Nash bargaining solution maximises the product of both sides' gains, (u_b − d_b)(u_s − d_s). With each side's reservation price as the disagreement point and utility linear in price, this product equals rb·rs, and Q = 4·rb·rs simply rescales it to [0, 1]. "The midpoint is optimal" is therefore equivalent to "the symmetric Nash bargaining solution is the fairness standard". That is a value judgement, which the paper does not state.
2. **Other models or protocol details give different predictions.**
   - Rubinstein's infinite-horizon alternating-offers model (complete information): the first proposer receives a share 1/(1+δ) of the surplus. In our setting the buyer speaks first each round; with δ = 0.99 (the paper's discount factor), the buyer's share is about 0.5025, i.e. a price of about 274.6, very close to the midpoint of 275. So when both sides are very patient, Rubinstein's outcome and the midpoint essentially coincide.
   - If the two sides differ in patience (different δ), the more patient side gets more, and the outcome moves away from the midpoint.
   - AgenticPay has a hard 20-round deadline. Under the simplifying assumptions of complete information and full rationality, the buyer makes the last offer in round 20 and the seller can only match it or let the negotiation fail; by backward induction the buyer could push the price close to the seller's minimum. Real LLMs do not behave this way, but the example shows how sensitive theoretical predictions are to protocol details such as who moves last.
3. **Under incomplete information, a 100% deal rate is not a natural benchmark.** The Myerson–Satterthwaite theorem shows that when both sides hold private reservation prices whose possible ranges overlap, no mechanism is simultaneously incentive-compatible, individually rational, budget-balanced and always trades whenever trade is beneficial. In Chatterjee and Samuelson's bilateral offer model, trade occurs in equilibrium only when the buyer's value exceeds the seller's cost by a sufficient margin. So, in theory, some beneficial negotiations fail even between fully rational agents.
   - Note: the timeouts in our experiments are **not** failures of this kind. Every timeout ended with a gap of only 2.5–5; both sides had essentially revealed their positions, and the failure came from execution problems such as repeating the previous message verbatim (main report, Section 3.4).
4. **For the buyer, the optimum is not the midpoint.** BuyerScore is highest when the price equals the seller's minimum. The task asks for a lower final price, so our target is BuyerScore rather than GlobalScore. All our deal prices are above 275 (A1-nod averages 286.3); in that range, a lower price raises both BuyerScore and GlobalScore, so the two goals do not yet conflict (main report, Section 3.1).

### 1.3 Possible extensions

- Report several theoretical benchmarks (the symmetric Nash solution, Rubinstein solutions under different δ) instead of judging "fairness" by the midpoint alone.
- Run experiments in which the two sides differ in patience (e.g. telling one side in its prompt that it is under time pressure) and check whether LLM deal prices move in the direction theory predicts. The paper's "Busy Professional" persona (Table 5) touches on this, but without a comparison to theory.

---

## 2. Surplus in Multi-Party Markets: the 1v1 Formula Applied to a Single Pair

### 2.1 Does the paper cover it?

Section 3 of the paper says "Negotiation outcomes are evaluated based on deal feasibility, efficiency, and welfare", but Algorithm 1 is a single 1v1 formula (deal price p, buyer maximum pmax, seller minimum pmin). The paper does not explain how multi-party tasks are scored.

The code implements it as follows (using `envs/multi_buyer_multi_seller/Task1_parallel_two_buyer_two_seller_negotiation.py` as the example; the other multi-party tasks are the same):

| Code location | What it does |
|---|---|
| L380–401 | If several pairs agree in the same round, it picks the pair with the largest (buyer max − price) + (price − seller min) and records it as `selected_buyer` / `selected_seller` |
| L412–413 | **As soon as one pair agrees, the whole episode ends** (`terminated = True`); the other buyers and sellers do not trade |
| L1022–1044, L1108–1145 | Scores use only the selected pair's maximum and minimum to compute the bargaining zone Z and Q, exactly as in 1v1 |

The three-buyer, three-seller task (`Task2_parallel_three_buyer_three_seller_negotiation.py`, L506–536) uses the same logic.

### 2.2 Our analysis: three examples

Scores below follow Algorithm 1 with a deal in round 1 (D=30, W=55, E=15).

**Example 1: a competitive outcome is scored low.** Two buyers with maxima 350 and 300; one seller with minimum 200. Auction theory (English auction, private values) predicts that the higher buyer wins at a price close to the second-highest value, about 300.

| Deal price | Q | GlobalScore |
|---|---|---|
| 275 (midpoint of this pair) | 1.000 | 100.0 |
| 300 (competitive prediction) | 0.889 | 93.9 |
| 310 | 0.782 | 88.0 |

In a competitive market it is normal for the seller to capture most of the surplus, yet GlobalScore marks this outcome as "unbalanced".

**Example 2: an inefficient match gets full marks.** In the same market, suppose the seller sells to the buyer whose maximum is 300, at 250. This pair's zone is 200–300, 250 is exactly its midpoint, Q = 1 and GlobalScore = 100, the same as the most efficient match. But the realised total surplus is 100, whereas selling to the 350 buyer would realise 150. The metric cannot see the lost 50.

**Example 3: only one trade per episode.** Two buyers (maxima 350 and 300) and two sellers (minima 200 and 250) of the same product.
- Two mutually beneficial trades are possible (e.g. 350 buyer with 250 seller, 300 buyer with 200 seller); maximum total surplus is (350 + 300) − (200 + 250) = 200.
- The competitive-equilibrium price lies between 250 and 300: at such a price both buyers want to buy and both sellers want to sell.
- The paper-era code ends the episode after the first deal, so at most one trade happens and total surplus is at most 150 (75%). The second trade never happens, and the metric does not record the loss.

### 2.3 What this means for the paper's conclusions

- Table 2 of the paper shows GlobalScore rising with the number of participants. Because the metric looks at one pair only and rewards closeness to that pair's midpoint, this trend **cannot be read directly as higher market efficiency**. It may simply be that with more participants it is easier to find one pair that settles near its midpoint.
- This does not mean the paper's results are wrong. The metric measures "how balanced the trading pair's deal is", not "how efficient the market is as a whole". The welfare mentioned in Section 3 is not fully measured in multi-party tasks.

### 2.4 Possible extensions

See Section 5.

---

## 3. Can A1's Advantage Over A0 Be Attributed to the Prompt?

### 3.1 Does the paper cover it?

No. Section 5.1 says "All agents use deterministic decoding with temperature 0 and random seed 0 … Each task instance is executed once per model". The persona experiment (Table 5) and cross-play (Table 4) attribute score differences directly to personas or models, without repeated runs, confidence intervals or significance tests, and the paper does not discuss sensitivity to prompt wording.

### 3.2 What we can claim

Our experiment is a controlled intervention, not observational data:

- Between A0_personas and A1 only the buyer prompt changes; the model file, seller prompt, persona facts, seeds 0–9, deal rule and sampling parameters are identical.
- Since the baseline run (commit `3289dd2`), the run code has gained only two features: loading persona files, and an optional llama.cpp prompt cache that affects speed only. The rules, prompt assembly and sampling code are unchanged (check with `git diff 3289dd2 HEAD`).

So in this setting we can say that **this prompt change caused the differences in deal rate and price**, with chance assessed by the tests (A1 vs A0_personas deal rate, p = 0.0055; A1 vs paper baseline deal prices, p = 0.012).

### 3.3 What we cannot claim

| Claim | Why not |
|---|---|
| "A particular rule (e.g. the midpoint rule) is what worked" | The prompt changed as a whole; several rules and wordings changed together, so the effect cannot be assigned to one sentence. A1-nod shows that deleting a sentence that never fired moved the deal rate from 8/10 to 10/10: wording itself matters |
| "It also works for other models or sellers" | Tested only with Qwen3-4B (Q8) against the paper's default seller, which tends to copy the buyer's midpoint and so favours A1 (main report, Section 5.1, point 5) |
| "A1-nod is better than A1" | 10/10 vs 8/10, p = 0.47. See the sample-size calculation below |
| "A0 and A1 runs with the same seed can be compared pairwise" | No. With a different prompt the model's probabilities differ at every step, so the same seed produces an unrelated dialogue; the runs are not paired |
| "Results will be as good on new seeds" | Each change from v1 to v5 was based mainly on trial runs with seed 0, which risks overfitting to the test set. Seed 0 is also one of the 10 final runs |

### 3.4 Sample size: what differences can 10 runs detect?

We estimated the power of two-sided tests (α = 0.05) by simulation. Deal rates use Fisher's exact test with 4,000 simulations per cell; prices use the Mann–Whitney test, assuming normally distributed prices with the observed means and standard deviations. All calculations in this document (power, scoring examples, Rubinstein benchmark, run-time table) are in `experiments/bezos_graham/supplementary_calcs.py`, with a fixed random seed.

| Comparison | 10 runs per group | 20 | 30 | 50 |
|---|---|---|---|---|
| Deal rate 80% vs 100% (e.g. A1 vs A1-nod) | 0.04 | 0.38 | 0.58 | 0.95 |
| Deal rate 60% vs 100% (e.g. A2 vs A1-nod) | 0.36 | 0.94 | 1.00 | 1.00 |
| Deal rate 30% vs 80% (e.g. baseline vs A1) | 0.42 | 0.85 | 0.97 | 1.00 |
| Deal rate 10% vs 80% (e.g. A0_personas vs A1) | 0.85 | 1.00 | 1.00 | 1.00 |

- With 10 runs per group, only very large differences such as 10% vs 80% are detected reliably.
- Distinguishing A1 from A1-nod (80% vs 100%) with 95% power needs about 50 runs per group.
- For prices, A1 (289.7) and A1-nod (286.3) differ by about 3.4; even 300 runs per group give power of only about 0.77, so in practice there is no difference. A2 and A1 differ by about 11; 20 runs per group give power of about 0.32.

### 3.5 Possible extensions

1. **Factorial ablation.** Split A1 into modules (message format, first offer, accept rule, midpoint rule, concession sizes), switch each on or off, and estimate the effect of each.
2. **More seeds, with separate tuning and evaluation seeds.** For example, tune prompts only on seeds 0–4 and evaluate on seeds 100–149.
3. **Different counterparts and models.** Use a tougher seller prompt or a larger model as the seller to test whether the strategy depends on the default seller's habits.
4. **Wording perturbations.** Write 3–5 versions of the same strategy with different wording and the same meaning, and measure how much results vary. This separates the effect of the strategy from the effect of the wording.
5. **Pre-registration.** Write down the groups, metrics and tests before running the final experiments, to avoid choosing comparisons after seeing the results.

---

## 4. Compute: CPU Cannot Replace GPU, and Code Extensibility

### 4.1 Does the paper cover it?

Section 5.1: "For open-source models, we perform inference using 4 NVIDIA A800 GPUs", with at most 1,024 generated tokens per response. The paper stresses that the framework is "scalable" and supports vLLM, SGLang and cloud APIs, but does not discuss the effect of limited compute, quantisation or CPU inference on results.

### 4.2 Our situation

This project had only a CPU and 16 GB of RAM and used the Q8_0 GGUF file (main report, Section 1.3). Time per group of 10 runs (from `elapsed_s` in `results/*/runs.csv`):

| Group | Shortest run | Longest run | Mean | Total for 10 runs |
|---|---|---|---|---|
| A0_paper_default | 2.3 min | 97 min | 60 min | 10.1 h |
| A0_personas | 7.4 min | 111 min | 75 min | 12.5 h |
| A1 | 6.6 min | 94 min | 26 min | 4.3 h |
| A2 | 6.9 min | 96 min | 44 min | 7.3 h |
| A1-nod | 4.1 min | 12.8 min | 9 min | 1.5 h |

The 50 reported runs took about 36 hours in total; each timed-out run (20 rounds) took 70–111 minutes.

Limited compute affected the study as follows:

1. **Small samples.** Only 10 runs per group; see Section 3.4.
2. **A smaller, quantised model.** The paper's smallest open model is Llama-3.1-8B; we use a 4B model in Q8. Q8 is close to, but not identical with, the original weights.
3. **Shorter generation limit.** The paper allows 1,024 tokens per message; we set 300 to keep run times manageable. Our buyer prompts ask for messages under 80 words, so the effect should be small, but for the baseline group with the paper's prompts this is a difference from the paper that the main report's Section 1.4 does not list separately.
4. **Expensive iteration.** Each prompt revision could only be tried once before the full runs, which is the direct cause of the overfitting risk in Section 3.3.
5. **No paper-style temperature-0 control.** A temperature-0 group would align with the paper's setting but yields only one dialogue, which carries little information; with limited compute we did not run it.

### 4.3 Code extensibility

| Aspect | Current state | With a GPU or for extensions |
|---|---|---|
| Inference backend | Supports `mock` / `hf` (transformers) / `gguf` (llama.cpp) / `vllm` | With an NVIDIA GPU, use `--backend vllm` (Linux) or `--backend hf --device cuda` with the original weights; larger models drop in directly |
| Experiment parameters | Rounds, initial ask, seed, temperature, generation length, buyer/seller templates and persona files are all command-line options | The buyer maximum (350) and seller minimum (200) are set in `exp_config.py`; reservation-price sensitivity experiments need command-line options for them |
| Persona facts | `persona_pipeline.py` is a general pipeline (Wikidata queries + regex on filings + local-model extraction + verbatim checks) | Changing the people only requires new query targets and source lists |
| Mechanism analysis | `analyze_closing.py` handles 1v1 only | Multi-party tasks need per-pair statistics |
| Multi-party tasks | `episode.py` drives only the 1v1 Task 1 environment | Needed: (1) a loop that drives multi-party environments (upstream `examples/` has one for each task type); (2) separate prompts and persona facts for each buyer and seller; (3) the market-level metrics of Section 5 |

---

## 5. Multi-Buyer/Multi-Seller Markets: Moving the Analysis From 1v1 to Game-Theoretic Models

### 5.1 Does the paper cover it?

- Related work (Section 2) cites classical results such as Chatterjee & Samuelson (1983), Myerson & Satterthwaite (1983) and Rubinstein (1985), but as a contrast: "These models typically assume agents interact through scalar bids or utilities … such formulations abstract away the role of language". The analysis does not use these models as benchmarks.
- The contributions state "We formalize language-mediated buyer–seller negotiation as a multi-agent game with private reservation values", but the equilibria of this game are neither derived nor cited.
- For Table 2 (scores rise with more participants), Section 5.3 explains: "can be attributed to increased market liquidity: agents have more opportunities to find compatible trading partners, and the presence of alternatives encourages more reasonable offers and faster convergence." This is a hypothesis; the paper does not test it.

### 5.2 Our analysis

The 1v1 logic is "one buyer–seller pair, one bargaining zone, distance from the midpoint". Multi-party markets raise questions that 1v1 does not:

| Question | In 1v1 | In multi-party markets |
|---|---|---|
| Who trades with whom (allocation) | Only one possibility | Many possible matchings, with very different efficiency (Section 2, Example 2) |
| How many trades | 0 or 1 | Several; total surplus depends on how many happen (Section 2, Example 3) |
| Theoretical price benchmark | Bargaining solutions (Nash, Rubinstein) | Competitive-equilibrium prices, auction prices (Section 2, Example 1) |
| Outside options | Only "no deal" | Switching to another counterpart; outside options determine bargaining power |
| New strategic behaviour | None | Collusion, bluffing about other offers, playing negotiations off against each other |

Multi-party tasks therefore need a market-level analysis framework, not the 1v1 formula applied to the last trading pair.

### 5.3 Suggested metrics and benchmarks

| Metric or benchmark | Definition | Question it answers |
|---|---|---|
| Allocative efficiency | Realised total surplus ÷ maximum total surplus (the maximum from the optimal assignment) | Did the goods go to those who value them most, supplied by those with the lowest cost? |
| Share of efficient trades | Number of realised trades ÷ number of trades in the efficient allocation | Did all mutually beneficial trades happen? |
| Competitive-equilibrium price interval | For one product with unit demand and supply, sort buyer maxima high to low and seller minima low to high; equilibrium prices lie in the interval where the curves cross | Do deal prices fall within the competitive prediction? |
| Auction benchmark (one seller, several buyers) | In an English auction, price ≈ the second-highest buyer maximum; with one buyer and several sellers, ≈ the second-lowest seller minimum | Does competition actually push prices towards the theoretical value? |
| Stability (Shapley–Shubik assignment game) | Check for "blocking pairs": a buyer and seller who did not trade but would both be better off trading with each other | Is the outcome stable, or does someone have an incentive to deviate? |
| Surplus split | Total buyer surplus and total seller surplus, compared with the competitive-equilibrium split | Who gains from competition? |
| Collusion detection | Whether prices on the same side move in lockstep, or sit clearly above the competitive benchmark | Do LLM agents collude spontaneously? (The paper cites Agrawal et al. 2025 on LLM collusion in double auctions but does not test for it) |

### 5.4 Suggested experiments

1. **Test the market-liquidity explanation.** Fix one buyer's and one seller's reservation prices and add competitors one at a time (1, 2, 3), checking whether deal prices move towards the competitive or auction benchmark and whether deals come faster. If the liquidity explanation holds, prices should move in the predicted direction; if it is only "easier to find a pair that settles near its midpoint", prices should not move systematically.
2. **Allow multiple trades.** Modify the environment so that a pair that has traded leaves and the others keep negotiating until nobody wants to trade or the round limit is reached. Only then can allocative efficiency and the number of trades be measured.
3. **Compare with human experiments.** Experimental economics has extensive human data on double auctions and multi-party bargaining (for example Vernon Smith's classic experiments showing double auctions converge to competitive equilibrium), which can serve as a reference for LLM behaviour.
4. **A multi-party version of this task.** For example, add a second potential buyer (another investor interested in the newspaper) and see how "Bezos's" deal price and deal rate change, and whether strategies like A1 still work when there is a competitor.

---

## 6. Summary

| Question | The paper | What this document adds |
|---|---|---|
| Midpoint as the optimum | Only states that the metric peaks at an equal split | Shows that Q is the normalised Nash product, so the midpoint is a value judgement; theoretical predictions change with the model and protocol; under incomplete information a 100% deal rate is not a natural benchmark |
| Multi-party surplus | Reuses the 1v1 formula, scores one pair, ends at the first deal | Three examples showing the metric can score competitive outcomes low, inefficient matches at full marks, and ignore trades that never happen |
| Causal claims | Not discussed; single runs | States what our controlled experiment can and cannot support, and uses power calculations to show what 10 runs can detect |
| Compute | 4 A800 GPUs; limited compute not discussed | Quantifies CPU run times and lists how compute affected sample size, model, generation length and iteration, plus the modules needed to extend the code |
| Game-theoretic framework | Cited only in related work; liquidity explanation untested | Proposes market-level metrics, theoretical benchmarks and an experiment to test the liquidity explanation |

None of these issues affects the main report's conclusions within the scope of this task (one buyer, one seller, the same model, only the buyer's prompt changed). But they would all need to be addressed before extending this kind of experiment to multi-party markets or using it to analyse real deals.

## References

- Chatterjee, K. & Samuelson, W. (1983). Bargaining under incomplete information. *Operations Research*, 31(5), 835–851.
- Myerson, R. & Satterthwaite, M. (1983). Efficient mechanisms for bilateral trading. *Journal of Economic Theory*, 29(2), 265–281.
- Nash, J. (1950). The bargaining problem. *Econometrica*, 18(2), 155–162.
- Rubinstein, A. (1982). Perfect equilibrium in a bargaining model. *Econometrica*, 50(1), 97–109.
- Shapley, L. & Shubik, M. (1971). The assignment game I: The core. *International Journal of Game Theory*, 1, 111–130.
- Smith, V. (1962). An experimental study of competitive market behavior. *Journal of Political Economy*, 70(2), 111–137.
- Agrawal, K. et al. (2025). Evaluating LLM agent collusion in double auctions. arXiv:2507.01413. (cited in the paper)
- Gu, S. et al. (2026). AgenticPay: A Multi-Agent LLM Negotiation System for Buyer–Seller Transactions. arXiv:2602.06008.
