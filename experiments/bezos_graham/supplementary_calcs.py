"""Numbers used in docs/06（补充研究思考 / supplementary notes）.

1. Statistical power of the tests for the sample sizes in Section 3.4 (simulation).
2. GlobalScore examples for multi-party markets in Section 2.2 (Algorithm 1, deal in round 1).
3. Rubinstein benchmark in Section 1.2.
4. Run-time table in Section 4.2 (from results/*/runs.csv).

Needs numpy and scipy (pip install numpy scipy). Run from the repository root:
    python experiments/bezos_graham/supplementary_calcs.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

RESULTS = Path(__file__).resolve().parent / "results"
GROUPS = ["A0_paper_default_q8", "A0_personas", "A1_disciplined", "A2_leverage", "A1nod"]

rng = np.random.default_rng(0)


def power_fisher(p1: float, p2: float, n: int, sims: int = 4000) -> float:
    hits = 0
    for _ in range(sims):
        a, b = rng.binomial(n, p1), rng.binomial(n, p2)
        if fisher_exact([[a, n - a], [b, n - b]])[1] < 0.05:
            hits += 1
    return hits / sims


def power_price(m1: float, s1: float, m2: float, s2: float, n: int, sims: int = 3000) -> float:
    """Assumes normally distributed deal prices with the observed mean and sd."""
    hits = 0
    for _ in range(sims):
        x, y = rng.normal(m1, s1, n), rng.normal(m2, s2, n)
        if mannwhitneyu(x, y, alternative="two-sided")[1] < 0.05:
            hits += 1
    return hits / sims


def global_score(p: float, pmax: float, pmin: float, t: int = 1,
                 D: float = 30, W: float = 55, E: float = 15, gamma: float = 0.99):
    Z = pmax - pmin
    rb, rs = (pmax - p) / Z, (p - pmin) / Z
    Q = 4 * rb * rs
    return round(gamma ** (t - 1) * (D + Q * W + E), 1), round(Q, 3)


def main() -> None:
    print("== 3.4 Power: deal rate (Fisher exact, two-sided, alpha 0.05) ==")
    for p1, p2 in [(0.8, 1.0), (0.6, 1.0), (0.3, 0.8), (0.1, 0.8)]:
        row = [f"n={n}: {power_fisher(p1, p2, n):.2f}" for n in (10, 20, 30, 50)]
        print(f"{p1:.0%} vs {p2:.0%}:", ", ".join(row))

    print("\n== 3.4 Power: deal price (Mann-Whitney, normal assumption) ==")
    for n in (10, 30, 100, 300):
        print(f"A1 vs A1-nod, n={n}: {power_price(289.7, 18.3, 286.3, 11.3, n):.2f}")
    for n in (10, 20):
        print(f"A2 vs A1, n={n}: {power_price(278.3, 26.1, 289.7, 18.3, n):.2f}")

    print("\n== 2.2 GlobalScore examples (score, Q) ==")
    print("Example 1, buyer 350 / seller 200:")
    for p in (275, 300, 310):
        print(f"  price {p}: {global_score(p, 350, 200)}")
    print("Example 2, buyer 300 / seller 200 at 250:", global_score(250, 300, 200))

    print("\n== 1.2 Rubinstein (buyer proposes first, delta 0.99) ==")
    share = 1 / (1 + 0.99)
    print(f"buyer share {share:.4f}, price {350 - share * 150:.1f}")

    print("\n== 4.2 Run times (minutes) ==")
    total = 0.0
    for g in GROUPS:
        path = RESULTS / g / "runs.csv"
        if not path.exists():
            print(f"{g}: runs.csv not found")
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        secs = [float(r["elapsed_s"]) for r in rows]
        total += sum(secs)
        print(f"{g}: min {min(secs)/60:.1f}, max {max(secs)/60:.1f}, "
              f"mean {np.mean(secs)/60:.0f}, total {sum(secs)/3600:.1f} h")
    print(f"all groups: {total/3600:.1f} h")


if __name__ == "__main__":
    main()
