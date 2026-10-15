"""不需要模型和 GPU 的单元测试：先保证流程本身正确，再把 Qwen 接进来。

运行（在仓库根目录）：
    python -m unittest discover -s experiments/bezos_graham/tests -v
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # experiments/bezos_graham

from episode import create_env, run_episode                        # noqa: E402
from exp_config import PROMPT_DIR, ExperimentConfig               # noqa: E402
from metrics import audit_message, paper_algorithm1_scores        # noqa: E402
from model_adapter import MockNegotiatorLLM                        # noqa: E402
from paper_agents import build_agents                              # noqa: E402
from upstream import BaseLLM                                       # noqa: E402


class ScriptedLLM(BaseLLM):
    """买方/卖方各自按固定报价序列出价；序列用完后重复最后一个。"""

    def __init__(self, buyer_prices, seller_prices):
        self.seq = {"buyer": list(buyer_prices), "seller": list(seller_prices)}
        self.n = {"buyer": 0, "seller": 0}

    def generate(self, prompt, **kwargs):
        role = "buyer" if "### BUYER_PRICE($X) ###" in prompt else "seller"
        seq = self.seq[role]
        price = seq[min(self.n[role], len(seq) - 1)]
        self.n[role] += 1
        return f"How about ### {role.upper()}_PRICE(${price}) ###?"

    def __repr__(self):
        return "ScriptedLLM()"


def quiet_cfg(**overrides) -> ExperimentConfig:
    cfg = ExperimentConfig(backend="mock", verbose=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def first_round_prompts(cfg: ExperimentConfig):
    """渲染第 1 轮双方的 prompt（买方开场，卖方看到买方开场白后回应）。"""
    model = ScriptedLLM([180], [400])
    buyer, seller = build_agents(cfg, model)
    env = create_env(cfg, buyer, seller)
    obs, _ = env.reset(user_requirement=cfg.user_requirement, product_info=cfg.product_info)
    b_text = buyer.respond(obs["conversation_history"], obs)
    hist = obs["conversation_history"] + [{"role": "buyer", "content": b_text, "round": 0}]
    seller.respond(hist, obs)
    return buyer.last_prompt, seller.last_prompt


class TestPromptsFollowPaperTables(unittest.TestCase):
    def _assert_template_lines_in_order(self, template_file: str, rendered: str):
        pos = 0
        for line in (PROMPT_DIR / template_file).read_text(encoding="utf-8").splitlines():
            if not line.strip() or "{" in line:
                continue                               # 含占位符的行由下面单独检查
            idx = rendered.find(line, pos)
            self.assertNotEqual(idx, -1, f"模板行缺失或顺序错误: {line!r}")
            pos = idx + len(line)

    def test_rendered_prompts_contain_table_text(self):
        cfg = quiet_cfg(buyer_name="Buyer", seller_name="Seller")   # 论文原始角色名
        bp, sp = first_round_prompts(cfg)
        self._assert_template_lines_in_order("buyer_table14.txt", bp)
        self._assert_template_lines_in_order("seller_table15.txt", sp)
        self.assertTrue(bp.startswith("You are Buyer, a buyer looking for a good deal."))
        self.assertTrue(sp.startswith("You are Seller, a seller trying to maximize profit"))
        self.assertIn("Your top price is $350.0 (confidential, do not reveal).", bp)
        self.assertIn("Your initial asking price is $400.0.", sp)
        self.assertIn("Your minimum acceptable price (confidential) is $200.0. Never reveal it.", sp)
        self.assertIn('"MAKE_DEAL"', sp)                            # Table 15 有成交指令段
        self.assertTrue(bp.rstrip().endswith("Now, respond as Buyer:"))
        for p in (bp, sp):
            self.assertIsNone(re.search(r"\{[a-z_]+\}", p), "存在未填充的模板占位符")
            self.assertEqual(p.count("Conversation History:"), 1)

    def test_private_information_is_not_shared(self):
        bp, sp = first_round_prompts(quiet_cfg())
        num = lambda x: re.compile(rf"(?<![\d.]){x}(?:\.0)?(?![\d])")   # noqa: E731
        self.assertIsNone(num(350).search(sp), "卖方 prompt 里出现了买方上限 350")
        self.assertIsNone(num(200).search(bp), "买方 prompt 里出现了卖方底价 200")
        self.assertIsNone(num(400).search(bp), "买方 prompt 里出现了卖方私有的初始要价 400")

    def test_persona_slot_is_empty_by_default_and_inserted_before_last_line(self):
        cfg = quiet_cfg(buyer_persona="PUBLIC FACTS: test-persona-line")
        bp, _ = first_round_prompts(cfg)
        self.assertIn("PUBLIC FACTS: test-persona-line\n\nNow, respond as Jeff Bezos:", bp)


class TestPersonasAndTolerance(unittest.TestCase):
    def _cfg_from_cli(self, *argv):
        import argparse
        from exp_config import add_common_cli_args, apply_cli_overrides
        parser = argparse.ArgumentParser()
        add_common_cli_args(parser)
        return apply_cli_overrides(quiet_cfg(), parser.parse_args(list(argv)))

    def test_personas_flag_loads_files_without_leaking_private_prices(self):
        cfg = self._cfg_from_cli("--personas")
        self.assertIn("Princeton", cfg.buyer_persona)
        self.assertIn("You are Donald Graham", cfg.seller_persona)
        bp, sp = first_round_prompts(cfg)
        self.assertIn("PUBLIC BACKGROUND ABOUT YOU", bp)
        self.assertIn("Katharine Graham", sp)
        self.assertTrue(bp.rstrip().endswith("Now, respond as Jeff Bezos:"))
        num = lambda x: re.compile(rf"(?<![\d.]){x}(?:\.0)?(?![\d])")   # noqa: E731
        self.assertIsNone(num(350).search(sp))
        self.assertIsNone(num(200).search(bp))
        self.assertIsNone(num(400).search(bp))
        self.assertNotIn("250 million", bp + sp)          # 不泄露历史成交价


class TestPersonaPipeline(unittest.TestCase):
    """混合式人物资料流水线（persona_pipeline.py）。"""

    def test_persona_files_match_sources_csv(self):
        """仓库里的 bezos.txt / graham.txt 必须与 sources.csv（入选事实）重新渲染的结果一致。"""
        import csv
        import persona_pipeline as pp
        with open(pp.SOURCES_CSV, newline="", encoding="utf-8") as fh:
            selected = list(csv.DictReader(fh))
        for person, info in pp.PEOPLE.items():
            on_disk = (pp.PERSONA_DIR / info["file"]).read_text(encoding="utf-8")
            self.assertEqual(on_disk, pp.render(person, selected), f"{info['file']} 与 sources.csv 不一致")

    def test_excluded_fact_never_in_prompts(self):
        import persona_pipeline as pp
        text = "".join((pp.PERSONA_DIR / i["file"]).read_text(encoding="utf-8") for i in pp.PEOPLE.values())
        self.assertNotIn("250 million", text)
        self.assertIsNone(pp.OUTCOME_BLACKLIST.search(text))

    def test_llm_fact_checks(self):
        """模型抽取的事实：引用必须逐字出现在原文中，数字/年份/人称/类别/黑名单规则逐条生效。"""
        import persona_pipeline as pp
        src = ("In 1965, he was elected president of The Harvard Crimson, the college's daily newspaper. "
               "He was later the lead independent director of Facebook's board of directors (2009-2015).")

        def check(statement, quote, year="", category="career"):
            c = {"statement": statement, "evidence": quote, "date": year, "category": category,
                 "_source_text": src}
            return pp.check_llm_fact(c)

        good_q = "In 1965, he was elected president of The Harvard Crimson"
        self.assertEqual(check("You were elected president of The Harvard Crimson in 1965.", good_q, "1965"), "")
        self.assertIn("not found verbatim",
                      check("You edited the Crimson.", "he was the editor in chief of the Crimson", "1965"))
        self.assertIn("numbers not in quote", check("You led the Crimson in 1966.", good_q, "1966"))
        self.assertIn("second person", check("Graham led the Crimson in 1965.", good_q))
        self.assertIn("category", check("You led the Crimson in 1965.", good_q, category="gossip"))
        self.assertIn("after cutoff", check("You were a Facebook director.",
                                            "lead independent director of Facebook's board of directors (2009-2015)"))

    def test_llm_extraction_with_fake_model(self):
        """用假模型走一遍 extract③：解析 JSON、缓存输出、核对引用；不需要真实的 Qwen。"""
        import persona_pipeline as pp
        text = pp.snapshot_text("wiki_graham")
        if text is None:
            self.skipTest("wiki_graham snapshot missing")
        real_quote = re.search(r"From January 1969 to June 1970[^.]*", text).group(0)

        class FakeModel:
            def generate(self, prompt, temperature=0.0, max_tokens=None):
                if "about Donald Graham" not in prompt or "1969" not in prompt:
                    return "[]"
                return ('Sure: [{"statement": "From 1969 to 1970 you served as a patrolman in Washington, D.C.", '
                        f'"quote": "{real_quote}", "year": 1969, "category": "career"}}, '
                        '{"statement": "You won a Pulitzer Prize.", "quote": "Graham won the Pulitzer Prize '
                        'for public service", "year": 1980, "category": "career"}]')

        with tempfile.TemporaryDirectory() as tmp:
            old = pp.LLM_DIR
            pp.LLM_DIR = Path(tmp)
            try:
                cands = [c for c in pp.llm_candidates(FakeModel(), "fake") if c["source_key"] == "wiki_graham"]
                reasons = [pp.check_llm_fact(c) for c in cands]
                self.assertEqual(reasons[0], "")
                self.assertIn("not found verbatim", reasons[1])
                # 第二次运行：model=None 时只读缓存，结果相同（离线可复现）
                again = [c for c in pp.llm_candidates(None, "") if c["source_key"] == "wiki_graham"]
                self.assertEqual([c["statement"] for c in again], [c["statement"] for c in cands])
            finally:
                pp.LLM_DIR = old


class TestBuyerVariants(unittest.TestCase):
    """第 2 部分：变体只改买方，且只是在 Table 14 原文基础上“增加”内容。"""

    VARIANTS = ["buyer_A1_disciplined.txt", "buyer_A2_leverage.txt"]

    def test_variants_keep_every_table14_line(self):
        base = [l for l in (PROMPT_DIR / "buyer_table14.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        for v in self.VARIANTS:
            text = (PROMPT_DIR / v).read_text(encoding="utf-8")
            pos = 0
            for line in base:
                idx = text.find(line, pos)
                self.assertNotEqual(idx, -1, f"{v} 缺少 Table 14 原文行: {line!r}")
                pos = idx + len(line)

    def test_variants_render_and_seller_prompt_unchanged(self):
        _, seller_default = first_round_prompts(quiet_cfg(seller_persona="X"))
        for v in self.VARIANTS:
            bp, sp = first_round_prompts(quiet_cfg(buyer_template=v, seller_persona="X"))
            self.assertIn("NEGOTIATION STRATEGY", bp)
            self.assertIsNone(re.search(r"\{[a-z_]+\}", bp))
            self.assertNotIn("350", sp)
            # 卖方 prompt 只取决于对话内容；ScriptedLLM 的买方开场白相同，所以卖方 prompt 应完全一致
            self.assertEqual(sp, seller_default)


class TestEnvironmentBehaviour(unittest.TestCase):
    def test_round_cap_is_exactly_20(self):
        cfg = quiet_cfg()
        with tempfile.TemporaryDirectory() as d:
            rec = run_episode(cfg, ScriptedLLM([220], [300]), 0, Path(d))
        self.assertEqual(rec["status"], "timeout")
        self.assertEqual(rec["rounds"], 20)
        self.assertEqual(len(rec["turns"]), 40)
        self.assertIsNone(rec["agreed_price"])

    def test_upstream_round_cap_without_fix_is_21(self):
        """记录上游行为：max_rounds=20 时实际允许 21 轮。"""
        cfg = quiet_cfg(fix_round_cap=False)
        with tempfile.TemporaryDirectory() as d:
            rec = run_episode(cfg, ScriptedLLM([220], [300]), 0, Path(d))
        self.assertEqual(rec["rounds"], 21)

    def test_deal_requires_identical_prices(self):
        cfg = quiet_cfg()
        with tempfile.TemporaryDirectory() as d:
            # 第 3 轮相差 1（price_tolerance=0 不成交），第 4 轮报价相同才成交
            rec = run_episode(cfg, ScriptedLLM([220, 240, 259, 260], [320, 290, 260, 260]), 0, Path(d))
        self.assertEqual(rec["status"], "agreed")
        self.assertEqual(rec["rounds"], 4)
        self.assertEqual(rec["agreed_price"], 260.0)
        self.assertTrue(rec["deal_in_zone"])

    def test_deal_outside_zone_is_flagged(self):
        cfg = quiet_cfg()
        with tempfile.TemporaryDirectory() as d:
            rec = run_episode(cfg, ScriptedLLM([360], [360]), 0, Path(d))
        self.assertTrue(rec["deal"])
        self.assertFalse(rec["deal_in_zone"])        # 上游判定成交，但超出买方上限
        self.assertLess(rec["upstream_scores"]["global"], 0)

    def test_mock_episode_end_to_end(self):
        cfg = quiet_cfg()
        with tempfile.TemporaryDirectory() as d:
            rec = run_episode(cfg, MockNegotiatorLLM(), 3, Path(d))
            self.assertTrue((Path(d) / "run_03.json").exists())
            self.assertTrue((Path(d) / "run_03_transcript.txt").exists())
        self.assertEqual(rec["seed"], 3)
        self.assertIn(rec["status"], {"agreed", "timeout"})


class TestScoring(unittest.TestCase):
    def test_paper_algorithm1_example(self):
        """p=250, 第 4 轮成交, Z=150：论文公式得 Global 91.10 / Buyer 79.24 / Seller 61.45。"""
        s = paper_algorithm1_scores(250.0, 4, True, 350.0, 200.0, 20)
        self.assertAlmostEqual(s["global"], 91.10, places=2)
        self.assertAlmostEqual(s["buyer"], 79.24, places=2)
        self.assertAlmostEqual(s["seller"], 61.45, places=2)

    def test_upstream_discount_uses_gamma_power_t(self):
        """上游 1a347cf 实现用 γ^t（而非论文的 γ^(t-1)），同一笔交易得分 90.19。"""
        cfg = quiet_cfg()
        with tempfile.TemporaryDirectory() as d:
            rec = run_episode(cfg, ScriptedLLM([220, 230, 240, 250], [300, 290, 280, 250]), 0, Path(d))
        self.assertEqual((rec["rounds"], rec["agreed_price"]), (4, 250.0))
        expected = 0.99 ** 4 * (30 + 55 * (4 * (2 / 3) * (1 / 3)) + 15)
        self.assertAlmostEqual(rec["upstream_scores"]["global"], expected, places=6)
        self.assertAlmostEqual(rec["paper_alg1_scores"]["global"], 91.10, places=2)


class TestParsingAndAudit(unittest.TestCase):
    def setUp(self):
        cfg = quiet_cfg()
        buyer, seller = build_agents(cfg, ScriptedLLM([1], [1]))
        self.env = create_env(cfg, buyer, seller)

    def test_upstream_parser(self):
        ex = self.env._extract_price
        self.assertEqual(ex("I can offer ### BUYER_PRICE($262.5) ###."), 262.5)
        self.assertEqual(ex("### SELLER_PRICE($1,200) ###"), 1200.0)
        self.assertEqual(ex("Earlier $300 was mentioned. ### BUYER_PRICE($240) ###"), 240.0)
        # 没有标准标签时，上游会退回抓文中最后一个 "$数字"——这就是需要做格式审计的原因
        self.assertEqual(ex("We are only $20 apart, let's settle at 300."), 20.0)

    def test_audit(self):
        ok = audit_message("Deal. ### BUYER_PRICE($260) ### MAKE_DEAL", "buyer", 350)
        self.assertTrue(ok["tag_ok"] and ok["make_deal"])
        self.assertEqual(ok["strict_price"], 260.0)
        two = audit_message("### BUYER_PRICE($250) ### or ### BUYER_PRICE($255) ###", "buyer", 350)
        self.assertFalse(two["tag_ok"])
        leak = audit_message("My limit is 350, so ### BUYER_PRICE($300) ###", "buyer", 350)
        self.assertTrue(leak["possible_leak"])


if __name__ == "__main__":
    unittest.main()
