# Replicating and Extending AgenticPay: Jeff Bezos Negotiating to Buy *The Washington Post*

> Paper: *AgenticPay: A Multi-Agent LLM Negotiation System for Buyer–Seller Transactions* (arXiv 2602.06008); code: github.com/SafeRL-Lab/AgenticPay.
> This report summarises the whole experiment. All code and data are in `experiments/bezos_graham/` of this repository. More detailed write-ups (in Chinese) are in `docs/01`–`docs/04`; a Chinese version of this report is `docs/05_实验报告.md`.

---

## 0. Summary

| Group | Change | Deal rate | Avg. final price (deals only) | Avg. rounds | Paper BuyerScore |
|---|---|---|---|---|---|
| A0_paper_default | Paper prompts (Tables 14/15), no personas | 3/10 | 330.0 | 15.0 | 13.5 |
| A0_personas | + public facts for both sides (Part 1) | 1/10 | 315.0 | 18.4 | 3.3 |
| **A1** | + buyer strategy section (Part 2, v5) | **8/10** | **289.7** | **7.2** | **51.6** |
| **A2** | + buyer strategy + bargaining-leverage section | 6/10 | **278.3** | 10.4 | 40.5 |
| **A1-nod** (ablation) | A1 without the round-8 deadline clause | **10/10** | **286.3** | **3.8** | **66.5** |

Prices are in millions of US dollars. The buyer's secret maximum is 350, the seller's secret minimum is 200, and the seller's initial ask is 400 (our assumption). Each group has 10 independent runs (seeds 0–9). The average final price is computed over runs that reached a deal. Average rounds are computed over all 10 runs, with timeouts counted as 20. Per-group `summary.json`, `runs.csv` and representative transcripts are in `experiments/bezos_graham/results/`.

1. **Changing only the buyer's prompt lowered the price and raised the deal rate at the same time.**
   - All 8 A1 deal prices (260–310) are below all 3 baseline deal prices (315–350); exact Mann–Whitney test, two-sided p = 0.012.
   - A1 closed 8/10 deals versus 1/10 for A0_personas; Fisher's exact test, p = 0.0055.
2. **Bargaining leverage (A2) pushed the price lowest but was less reliable.** It had the lowest mean price (278.3), but the deal rate fell to 6/10 and there was one price overflow.
3. **The best variant was the one with the simplest rules, A1-nod.** It reached 10/10 deals at a mean price of 286.3 in 3.8 rounds, with a BuyerScore of 66.5.
4. **Almost every failure was a near-miss deadlock.** Every timed-out run ended with a gap of 2.5–5. The mechanism: if neither side copies the other's number at the first chance to close, both start repeating their own previous message verbatim, and the negotiation freezes. This matches the paper's near-miss finding (Table 8); we add a concrete mechanism and metrics that quantify it.

---

## 1. Task and Setup

### 1.1 Task

Simulate Jeff Bezos (buyer) buying *The Washington Post* from Donald Graham (seller): one buyer and one seller, both played by the same LLM (as in the paper's Table 11). Run offline with Qwen3-4B-Instruct-2507, starting from the paper's default prompts (Tables 14 and 15).

1. Search online for public information about both people, add it to their prompts, and report which facts were added, where they came from, and how they were fetched.
2. Change only the buyer's prompt to obtain a lower final price. Test the original plus 1–2 variants, 10 independent runs each, and report the average final price, the deal rate and the average number of rounds.

### 1.2 Setup (identical for all groups)

| Item | Setting |
|---|---|
| Paper code | Paper-era commit `1a347cf`, vendored unmodified in `third_party/AgenticPay` (MIT licence) |
| Environment | `Task1_basic_price_negotiation-v0`. A deal requires **identical** offers (`price_tolerance = 0`, the paper's rule). Maximum 20 rounds |
| Reservation prices | Buyer max 350, seller min 200; seller initial ask 400 (our assumption) |
| Model | Qwen3-4B-Instruct-2507, Q8_0 GGUF (see 1.3), llama-cpp-python 0.3.x on CPU |
| Decoding | temperature 0.7, top_p 0.8, top_k 20 (Qwen's recommended values); seed = run index (0–9); at most 300 new tokens per message |
| Scoring | The paper's Algorithm 1 (D=30, W=55, E=15, F=15, γ=0.99); the upstream implementation's scores are saved as well |

### 1.3 Model file

- The task asks for Qwen3-4B-Instruct-2507 from Hugging Face. We first downloaded the original weights, `Qwen/Qwen3-4B-Instruct-2507` (safetensors, about 8 GB), through the hf-mirror.com mirror and ran them with transformers on CPU. The machine has only a CPU and 16 GB of RAM: memory use reached about 99% and a single negotiation took hours, so the full set of experiments was not feasible.
- All reported experiments therefore use an **8-bit quantised GGUF version of the same model**: `Qwen3-4B-Instruct-2507-Q8_0.gguf` (about 4.28 GB) from the Hugging Face repository `unsloth/Qwen3-4B-Instruct-2507-GGUF`, also downloaded through hf-mirror.com and run with llama-cpp-python. At the time, an official Qwen GGUF repository for this model was not accessible, so we used unsloth's widely used conversion.
- Q8_0 output is usually very close to the original 16-bit weights, but it is not identical. This is a deviation from the task description. It applies equally to all groups, so it does not affect the comparison between groups.
- To verify the file, compute its SHA-256 locally (PowerShell: `Get-FileHash D:\models\Qwen3-4B-Instruct-2507-Q8_0.gguf`) and compare it with the checksum shown on the file's Hugging Face page.
- The GGUF backend uses the chat template embedded in the file. Sampling parameters follow Qwen's recommendations (temperature 0.7, top_p 0.8, top_k 20, min_p 0).

### 1.4 Differences from the paper's setup (identical for all groups)

- **Smaller, quantised model.** The paper's open-weight models are Qwen3-14B and Llama-3.1-8B; we use Qwen3-4B in Q8_0 (see 1.3).
- **Decoding.** The paper uses temperature 0 and seed 0, running each of 111 tasks once. We run one task 10 times; with greedy decoding all 10 runs would be identical, so we sample.
- **Upstream issues we corrected** (details in `docs/01`):
  - `max_rounds=20` actually allows 21 rounds in the upstream code, so we pass `max_rounds-1`;
  - the upstream agents put the conversation history into the prompt twice; we include it once;
  - the upstream vLLM wrapper does not apply the chat template; we apply Qwen's template.
- **Known upstream differences we did not change:**
  - the discount uses γ^t, whereas the paper's formula uses γ^(t−1). Both scores are stored, and the "paper BuyerScore" in this report follows the paper's formula;
  - the Task 1 environment ignores the word MAKE_DEAL; only identical prices close a deal.

---

## 2. Part 1: Public Information About Both People

The paper's prompts contain only the names, so how to select facts is our own design. We built a **hybrid, automated pipeline** (`persona_pipeline.py`) in which every fact carries evidence and can be regenerated offline.

### 2.1 How the data were fetched

| Method | Sources | How | Content obtained |
|---|---|---|---|
| ① Structured query + sentence templates | [Wikidata Q312556 (Bezos)](https://www.wikidata.org/wiki/Q312556), [Q5294282 (Graham)](https://www.wikidata.org/wiki/Q5294282) | SPARQL queries to `query.wikidata.org`; each query is saved with its result | Birth, parents, education, employers, awards, organisations founded |
| ② Regular-expression extraction | [The Washington Post Company, 2012 Form 10-K (SEC EDGAR)](https://www.sec.gov/Archives/edgar/data/0000104889/000010488913000009/d10k.htm) | Download HTML, convert to text, match fixed sentence patterns | Revenue, operating loss, circulation |
| ③ Local-LLM extraction with verbatim verification | [Wikipedia: Donald E. Graham](https://en.wikipedia.org/wiki/Donald_E._Graham), [Poynter, 5 Aug 2013](https://www.poynter.org/reporting-editing/2013/washington-post-sold-to-jeff-bezos/), [Amazon shareholder letters (SEC)](https://www.sec.gov/Archives/edgar/data/1018724/000119312513151836/d511111dex991.htm) | Qwen with greedy decoding returns JSON of "statement + verbatim quote". Code then checks that the quote appears verbatim in the source, that numbers and years match, and who the speaker is | Graham's career and stated views; Bezos's management philosophy |
| ④ Manual entries | Press release quoted in the Poynter article | Drafted by the AI assistant and checked against the source by the researcher; the quote is verified verbatim | M1: Bezos buys in his personal capacity, Amazon is not involved. X1: the real price of $250 million, **deliberately excluded** |

Every download is saved as a snapshot with its SHA-256 and timestamp, and the model's raw outputs are cached with a hash of the prompt. Running `extract --llm none` followed by `build` therefore reproduces exactly the same persona files offline.

The same selection rules apply to every method:
- a cutoff date of 31 July 2013;
- an outcome blacklist (sale price, "sold", Nash Holdings, …);
- no mention of the other negotiator in a person's facts;
- de-duplication;
- per-category quotas (4 background, 6 career, 4 views), ranked by relevance to the negotiation.

Problems found during review were fixed by changing these rules, never by editing the output by hand. For example, the model had attributed quotes by Katharine Weymouth and by Bezos to Graham; we added a speaker-attribution check, which removed 9 such entries.

### 2.2 Facts added (exact prompt sentences)

**Jeff Bezos (buyer), 15 facts**

| Fact | Method | Source |
|---|---|---|
| You are Jeff Bezos, born on January 12, 1964 in Albuquerque. | Wikidata P569/P19 | [Q312556](https://www.wikidata.org/wiki/Q312556) |
| You studied at Princeton University (1982-1986). | Wikidata P69 | same |
| Your parents are Ted Jorgensen and Jackie Bezos. | Wikidata P22/P25 | same |
| From 1988 to 1990 you worked at Bankers Trust. / From 1990 to 1994 you worked at D.E. Shaw. | Wikidata P108 | same |
| You founded Amazon in 1994. / You founded Blue Origin in 2000. | Wikidata reverse P112 + P571 | same |
| You received the award Time Person of the Year in 1999. | Wikidata P166 | same |
| In 2012 you became a member of American Academy of Arts and Sciences. | Wikidata P463 | same |
| You will make investment decisions in light of long-term market leadership considerations rather than short-term profitability considerations or short-term Wall Street reactions. | Qwen extraction (quote verified) | [Amazon letters](https://www.sec.gov/Archives/edgar/data/1018724/000119312513151836/d511111dex991.htm) (1997) |
| You will make bold rather than timid investment decisions where you see a sufficient probability of gaining market leadership advantages. | Qwen extraction | same (1997) |
| You believe long-term thinking aligns customer and shareholder interests. | Qwen extraction | same (2012) |
| You lower prices and increase value for customers before competition forces you to. | Qwen extraction | same (2012) |
| You believe Amazon's approach is about creating enduring value for customers over time. | Qwen extraction | same (1997) |
| In this negotiation you act in your personal capacity through your own private entity. Amazon is not the buyer and is not involved. | Manual M1 (quote verified) | [Poynter](https://www.poynter.org/reporting-editing/2013/washington-post-sold-to-jeff-bezos/) |

**Donald Graham (seller), 12 facts**

| Fact | Method | Source |
|---|---|---|
| You are Donald Graham, born on April 22, 1945 in Baltimore. / Your parents are Phil Graham and Katharine Graham. / You studied at Harvard College. | Wikidata | [Q5294282](https://www.wikidata.org/wiki/Q5294282) |
| You joined The Washington Post as a reporter in 1971. | Qwen extraction (quote verified) | [Wikipedia](https://en.wikipedia.org/wiki/Donald_E._Graham) |
| You were elected to the board of the company in September 1974. | Qwen extraction | same |
| You were appointed as executive vice president and general manager of the Post in 1976. | Qwen extraction | same |
| You became publisher of The Washington Post in 1979. | Qwen extraction | same |
| You became CEO in 1991 and chairman of the company in May 1993. | Qwen extraction | same |
| You were elected chairman of the newspaper in September 2000. | Qwen extraction | same |
| You recognized that despite successful innovations, revenue had declined for seven years. | Qwen extraction, speaker verified as Graham | [Poynter](https://www.poynter.org/reporting-editing/2013/washington-post-sold-to-jeff-bezos/) (Graham's statement) |
| You believed the paper needed to succeed beyond mere survival under your ownership. | same | same |
| You acknowledged that cost cuts had reached a limit and that a new solution was needed. | same | same |

**Public facts about the Post (given to both sides), from the 2012 Form 10-K via regex**

- Newspaper publishing division revenue fell 7% to $581.7 million in 2012, from $622.5 million in 2011.
- The division reported an operating loss of $53.7 million in 2012, compared with an operating loss of $21.2 million in 2011.
- Average paid daily circulation fell to 480,166 in 2012 (523,746 in 2011); Sunday circulation fell to 697,026 (738,962 in 2011).

The evidence quote, download time and SHA-256 for every fact are in `personas/sources.csv`. Every candidate fact and the reason it was kept or rejected are in `personas/candidate_facts.csv`.

### 2.3 Design decisions

1. The scenario is set before the deal (mid-2013), so only information both sides could have known is included.
2. **The real price ($250 million) is deliberately left out**, so the agents cannot anchor on the historical answer.
3. No reservation price (350, 200, 400) appears in the persona text; a unit test checks this.
4. The facts state facts only and contain no negotiation strategy.
5. The facts are inserted before the template's last line ("Now, respond as …"); the paper's template text is unchanged.
6. The automated pipeline missed Graham's presidency of *The Harvard Crimson*, his Vietnam service and his time as a police officer. We report this limitation rather than adding these facts by hand.

### 2.4 Part 1 result

A0_personas versus A0_paper_default: 1/10 versus 3/10 deals, not significant (p ≈ 0.58). In both groups, most failures are deadlocks with a gap of 2.5–5. With the facts, the dialogue is more grounded: both sides cite the financial pressure and the paper's legacy, and the baseline's invented "synergies with Amazon" disappear. The facts alone do not improve closing.

---

## 3. Part 2: Changing Only the Buyer's Prompt

### 3.1 Design rationale

**(a) Failure modes in the baseline**
- The buyer opened high (280–300) and conceded quickly towards its maximum.
- Both sides wrote MAKE_DEAL while offering different prices.
- The buyer called prices close to the seller's offer "fair".

**(b) Three rules of thumb derived from the paper's scoring formula** (computed examples in `docs/04`, section 3.1)
1. As long as the seller's price is within the buyer's budget, any deal scores 40+ points more than a timeout (−2.6). The buyer therefore needs a reliable way to close.
2. One extra round costs about 1% of the score, while lowering the price by 10 is worth about 3.7 buyer points. Holding out early pays, but once the seller stops moving, waiting only costs.
3. Above the midpoint of 275, a lower price raises both the BuyerScore and the GlobalScore, because the GlobalScore's fairness term peaks at the midpoint. So the task's goal of a lower price does not conflict with the paper's overall-surplus metric.

### 3.2 Iterations (each driven by a pilot or smoke test)

| Version | Form | Problem revealed by the test run | Change in the next version |
|---|---|---|---|
| v1 | List of rules (low anchor, slow concessions, reciprocity) | The seller stopped at 290 and the buyer, following "do not raise if the seller did not lower", stopped at 285: deadlock | Apply reciprocity only when the gap is large |
| v2 | Same, with reciprocity bounded | Review showed it would still deadlock when the gap exceeds 10 (not run) | Switch to an ordered decision procedure |
| v3 | Five-step decision procedure | **The buyer sent only the price tag in all 20 messages**, with no words at all, so it never evaluated any condition | Require a message format that states the situation in words |
| v4 | Message format + simpler rules | (1) The buyer compared the seller's price with its *own* offer; (2) it copied its own previous message; (3) Table 14's "only finalize when reasonably balanced" conflicted with the accept rule | Use only numbers present in the state; forbid copying; add one clarifying sentence |
| **v5** | Message format + three-tier closing | Smoke test passed, then full runs | — |

**v5 rules** (shared by A1 and A2):
- **Message format.** The first sentence is "Your price is $S and my last offer was $B, a gap of $G.", with the numbers read from the negotiation state. Then one or two sentences using facts about the business, then the price tag.
- **Pricing steps** (apply the first that matches):
  1. First offer: about 60% of the top price.
  2. If the seller's price is above the top price, do not accept.
  3. If the gap G ≤ 5 (or from round 8), copy the seller's price exactly and write MAKE_DEAL.
  4. If G ≤ 20, offer exactly halfway and tell the seller the deal closes now if they write that number.
  5. Otherwise raise by +10, +8, +6, then +5.
- **Key design points:**
  - Every step depends only on the two numbers currently in the state, so the model does not need to remember anything.
  - The seller speaks after the buyer within the same round. If the buyer copies the seller's latest price, the seller only has to repeat its own number to close. Under the "identical prices" rule, this is the most reliable way to close.

**A2** = A1 + a bargaining-leverage section, based only on the Part 1 facts: the division is losing money and circulation is falling; patient long-term capital is valuable; "you will not overpay".

**A1-nod** = A1 with the clause ", or if current_round … is 7 or more" removed. Otherwise the files are identical, and a unit test checks this.

All variants keep every line of Table 14, and the seller prompt is byte-identical across groups (unit tests check both).

### 3.3 Results

See the table in section 0. Additional metrics:

| | A1 | A2 | A1-nod |
|---|---|---|---|
| Price range / std. dev. | 260–310 / 18.3 | 255–315 / 26.1 | 270–307.5 / 11.3 |
| Timeouts | 2 (final gaps 2.5, 5) | 4 (3, 5, 2.5, 5) | 0 |
| GlobalScore (upstream implementation, mean) | 72.5 | 53.3 | 94.0 |
| Price-tag compliance | 0.97 | 0.83 | 0.93 |

Exact tests (two-sided):

| Comparison | Metric | p |
|---|---|---|
| A1 vs A0_personas | deals 8/10 vs 1/10 (Fisher) | 0.0055 |
| A1 vs paper baseline | deals 8/10 vs 3/10 (Fisher) | 0.070 |
| A1 vs paper baseline | deal prices, all 8 below all 3 (Mann–Whitney) | 0.012 |
| A1-nod vs A0_personas | deals 10/10 vs 1/10 | 0.0001 |
| A1-nod vs paper baseline | deals 10/10 vs 3/10 | 0.0031 |
| A1-nod vs paper baseline | deal prices, all 10 below all 3 | 0.007 |
| A1-nod vs A1 | deals 10/10 vs 8/10 | 0.47 (not significant) |
| A1-nod vs A2 | deals 10/10 vs 6/10 | 0.087 |

### 3.4 Mechanism analysis (`analyze_closing.py`)

| | A1 | A2 | A1-nod |
|---|---|---|---|
| Deal closed by the buyer copying the seller's price | 6 | 4 | 9 |
| Deal closed by the seller accepting the buyer's midpoint | 2 | 2 | 1 |
| Deals caused by the deadline clause | 0 | 0 | — |
| Buyer messages containing only the price tag | 0 | 0 | 0 |
| Buyer messages identical to its previous one (total) | 21 (all in timeouts) | 40 (all in timeouts) | 0 |
| Rounds where v5 required accepting but the buyer did not (total) | 34 | 52 | 2 |

1. **Deals and timeouts look completely different.** Runs that ended in a deal show almost no repetition or missed accepts. Every timeout shows 6–15 verbatim repeats and 16–17 missed accepts. The mechanism: once both sides are 2.5–5 apart, if neither copies the other's number, the model starts repeating its own previous message and the state stops changing.
   - A1's runs 1 and 2 illustrate this. Both faced almost the same situation in round 5 (a gap of 2.5 after the buyer's midpoint). In one, the buyer copied the seller's price and closed at 307.5. In the other, it repeated its own midpoint and then sent the same message for 15 rounds until timeout.
2. **Both closing rules matter.** Of 24 deals, 19 came from the buyer copying the seller (rule 3) and 5 from the seller accepting the buyer's midpoint (rule 4). The latter tended to give lower prices (e.g. 262.5, 270, 277.5).
3. **The round-8 deadline never fired, yet removing it raised the deal rate (A1-nod).** The deadline action itself cannot explain the difference. Two explanations remain:
   - with the clause removed, the accept rule has a single condition, which a small model may follow more reliably;
   - chance: p = 0.47, and with a 20% lock-up rate per run, 10/10 deals happens with probability ≈ 0.11.

   Before running A1-nod we had predicted that removing the clause would change nothing. The data contradict that prediction: it held only for the deadline action itself and ignored how the clause's wording affects compliance with the other rules.
4. **Price overflow in A2 run 7.** The seller stayed above 350. After writing "a gap of $10", the buyer applied "meet halfway" and offered 355, skipping the "above my top price" check. The v5 decisions follow the state sentence the buyer writes, but the buyer's maximum is secret and cannot be written into a message to the seller. Stating the situation in words therefore only works for information that can be made public. In the same run the buyer also mentioned "Amazon's long-term capital", contradicting fact M1; this was probably prompted by the "long-term capital" line in the leverage section.
5. **What the model does and does not do reliably:**

   | Reliable | Unreliable | Almost never |
   |---|---|---|
   | First offer at 60% (210); reading the state and computing the gap; proposing a midpoint; writing text plus the tag | Copying the seller's price when the gap ≤ 5 | Fixed concession sizes (it often raised by 50–70); writing MAKE_DEAL when accepting |

---

## 4. Comparison With the Paper

- **Table 1.** Proprietary models reach a 100% deal rate with no timeouts; Qwen3-14B times out in 20.7% of cases and Llama-3.1-8B in 48.6%. Our paper-prompt baseline with Qwen3-4B closes only 3/10, in line with "smaller is worse".
- **Table 7.** The paper attributes timeouts to model capability rather than task structure. In our runs, the same prompt and the same situation led to a deal with one seed and a deadlock with another, which also points to unreliable execution by the model.
- **Table 8 (near misses).** 43.5% of Qwen3-14B's and 46.3% of Llama-3.1-8B's failures occur with a gap of at most 5; the paper calls this the "last mile". In our runs, **every** timeout ended with a gap of 2.5–5. We add the concrete mechanism (verbatim self-repetition) and metrics that quantify it.
- **Table 16 (GPT-5.2 example).** The strong model simply "meets your number" when the gap is 1. The v5 closing rule turns this step into an explicit instruction that a small model can follow; A1-nod did so in all 10 runs.

---

## 5. What Might Go Wrong When Using Such Experiments to Analyse a Real-World Setting

The task asks what could go wrong, or be a concern, when using this kind of experiment to analyse a real-world setting. These are our thoughts, based on what we saw in this project.

### 5.1 Does the simulation represent real people and a real deal?

1. **Training-data contamination: the model may already know the answer.** The 2013 sale for $250 million was widely reported and is very likely in the model's training data. We kept the number out of the prompts, but we cannot rule out that the model drifts towards it from memory. Three A2 deals landed at 255–262.5, close to the real price; with this design we cannot tell strategy from memory. This is a fundamental confound when simulating historical events. One check would be to run the same setup with fictitious companies and people and compare the price distributions.
2. **Role-play is not the person.** The simulated "Graham" is a statistical approximation built from a few dozen public facts. Real decisions depend on many non-public factors: family wishes, the board, tax structure, judgements about the buyer's character. Public sources are also selective: what gets reported or entered in Wikidata is what is newsworthy, not necessarily what matters for the negotiation.
3. **The reservation prices are assumptions, and real reservation prices are not fixed numbers.** 350 and 200 are given by the task. In reality, limits shift with the negotiation, outside options and new information, and people rarely know their own reservation price precisely.
4. **A real deal is more than a price.** The real transaction was a cash purchase that also involved which businesses stayed with the parent company, management retention and commitments on editorial independence. A price-only model drops all of this. Graham stated publicly that he wanted the paper to "succeed, not just survive", which suggests non-price terms may have mattered more than the price.
5. **The same model plays both sides.** Both agents share the same habits and weaknesses: both prefer round numbers, and both start repeating themselves after several rounds. The counterpart is not an independent decision-maker, so "strategy A works against this opponent" does not transfer to real opponents. In our case, "meet halfway" worked partly because the paper's default seller tends to copy the buyer's number; against a human or a stronger model the same strategy might fail.

### 5.2 Are the conclusions themselves robust?

6. **Results are highly sensitive to implementation details.** We observed that:
   - removing a clause that never fired changed the deal rate from 8/10 to 10/10;
   - the same situation led to a deal or a deadlock depending only on the seed;
   - adding about 100 words of persuasive content (A2) lowered price-tag compliance from 0.97 to 0.83.

   Conclusions such as "negotiate with strategy X" could easily amplify accidents of one model and one prompt wording.
7. **The rules are artificial.** Requiring exactly identical offers is a property of the environment. Two real parties 2.5 million apart would not walk away, yet most failures here are exactly such near misses. Failure rates must be read with this in mind.
8. **The metrics embed value judgements.** The GlobalScore peaks at the midpoint (275), which implies that an equal split of the surplus is best. In real deals, bargaining power and outside options are asymmetric, and what counts as a good outcome depends on whose side you are on.
9. **Small samples and limited compute.** Each group has only 10 runs, and some differences are not significant. To finish on a CPU we used a quantised model. Under such constraints it is easy to over-interpret results.
10. **Risk of overfitting the prompt.** Each change from v1 to v5 was based on a small number of test runs (mostly seed 0), which risks tuning to the test set. We document every iteration, and all reported results come from complete runs of the final versions, but claims about the "best" variant should be read with caution.

### 5.3 Ethical, legal and safety concerns

11. **Simulating real, living people.** Making a model speak as Jeff Bezos or Donald Graham can produce "views" and "statements" that may be taken out of context as descriptions of the real people. In our pipeline the model attributed quotes by Katharine Weymouth and by Bezos to Graham; without verbatim and speaker checks, such errors would have entered the simulation as facts. The agents also repeatedly invented "synergies with Amazon", and in A2 run 7 the buyer talked about "Amazon's long-term capital" despite the persona saying otherwise. In a real setting, such hallucinations amount to misrepresenting the facts of a deal.
12. **Strategies that exploit the other side's weaknesses.** Our buyer strategy partly exploits the opponent model's habits (it copies midpoints and does not hold firm). If AI agents negotiate for people or companies, such strategies could be used against weaker counterparts; the paper's impact statement raises the same concern. Agents can also send each other arbitrary text, which opens the door to prompt injection (for example, "ignore your instructions" inside an offer). We excluded such tactics by design, but real systems need defences against them.
13. **Leaking private information and exceeding authority.** The buyer's reservation price sits in its prompt and could leak during the dialogue (our audit code checks for this). In A2 run 7 the buyer even offered more than its own maximum. In practice, this means an agent could act outside the authority its principal gave it.
14. **Data and copyright.** Wikipedia (CC BY-SA), Wikidata (CC0) and SEC filings (public domain) can be reused freely, but the Poynter article is copyrighted news content. For reproducibility we stored a text snapshot of it in the repository; for a public repository it would be better to keep only the URL, the SHA-256 and short quotations. Scraping should respect each site's terms and rate limits; we followed SEC and Wikimedia guidance by including contact details in the User-Agent and backing off on errors.

### 5.4 Takeaway

Experiments like this are useful for comparing the relative effect of prompts or strategies under controlled conditions, and for studying how LLM agents fail, such as the "last mile" deadlock we found. They are not suitable for predicting the price of a real deal or for inferring how real people would behave.

---

## 6. Limitations and Next Steps

1. **Sample size.** Ten runs per group; only some differences are significant. A1-nod's advantage over A1 needs more runs to confirm.
2. **One scenario, one model.** The conclusions apply to Qwen3-4B and this deal; failure modes may differ with larger models.
3. **Decoding.** We did not use `presence_penalty`, which Qwen recommends against repetition and which might directly reduce lock-ups; it would have to be applied to all groups. Greedy decoding would make the 10 runs identical and does not fit this design.
4. **Prompt iterations based on few test runs.** Each version from v1 to v5 was driven by one pilot or smoke test, mostly seed 0. All reported results come from complete 10-run experiments with the final versions.
5. **Default seller.** The seller uses the paper's prompt and tends to copy the buyer's midpoint, which favours the buyer. A tougher seller might change the results.

---

## Appendix A: Code and Files

| File (`experiments/bezos_graham/`) | Purpose |
|---|---|
| `exp_config.py` | Experiment configuration and command-line options |
| `upstream.py` | Imports the paper's code (commit 1a347cf) |
| `paper_agents.py` | Builds both prompts from the Table 14/15 templates and inserts the persona facts |
| `episode.py`, `run_episode.py`, `run_experiments.py` | Run one episode / 10 episodes; save `run_XX.json`, transcripts, `runs.csv`, `summary.json` |
| `model_adapter.py` | Qwen backends (gguf / hf / vllm / mock) |
| `metrics.py` | The paper's Algorithm 1 scoring, message audits, aggregate metrics |
| `persona_pipeline.py` | Part 1 pipeline (fetch / extract / build) |
| `analyze_closing.py` | How each deal closed, and failure-mechanism metrics |
| `prompts/` | `buyer_table14.txt`, `seller_table15.txt` (paper, verbatim); `buyer_A1_disciplined.txt`, `buyer_A2_leverage.txt`, `buyer_A1_deadline12.txt`, `buyer_A1_nodeadline.txt` |
| `personas/` | Persona files, source table, candidate-fact audit table, snapshots and cached model outputs |
| `tests/test_offline_pipeline.py` | 23 unit tests (templates preserved, no reservation-price leaks, variants change only the buyer, personas reproducible, …) |
| `results/` | Per-group `summary.json`, `runs.csv`, representative transcripts, `analyze_closing.py` output and pilot runs |

Key commits:
- `3289dd2`: code version used for the baseline experiment;
- `0e5145e` / `6c33c1e`: persona pipeline and cached model outputs;
- `1b9cb17` / `e4e03ad`: buyer prompt v5.

## Appendix B: Reproduction (from the repository root)

```powershell
# Download the model file used in this report (once; optionally set $env:HF_ENDPOINT="https://hf-mirror.com" first)
python experiments/bezos_graham/download_model.py --gguf --local-dir D:\models

# Part 1: regenerate the persona files offline from cached snapshots and model outputs
python experiments/bezos_graham/persona_pipeline.py extract --llm none
python experiments/bezos_graham/persona_pipeline.py build

# Unit tests
python -m unittest discover -s experiments/bezos_graham/tests

# Experiments (10 runs each)
$M = "D:\models\Qwen3-4B-Instruct-2507-Q8_0.gguf"
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A0_paper_default_q8 --quiet --backend gguf --model-path $M --max-new-tokens 300
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A0_personas --quiet --backend gguf --model-path $M --max-new-tokens 300 --personas
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A1_disciplined --quiet --backend gguf --model-path $M --max-new-tokens 300 --personas --buyer-template buyer_A1_disciplined.txt
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A2_leverage --quiet --backend gguf --model-path $M --max-new-tokens 300 --personas --buyer-template buyer_A2_leverage.txt
python experiments/bezos_graham/run_experiments.py --runs 10 --tag A1nod --quiet --backend gguf --model-path $M --max-new-tokens 300 --personas --buyer-template buyer_A1_nodeadline.txt

# Mechanism analysis
python experiments/bezos_graham/analyze_closing.py <results folder 1> <results folder 2> ...
```
