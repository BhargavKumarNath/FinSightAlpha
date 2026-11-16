"""Unit tests for the Phase 3 evaluation harness (no network, no models)."""

import numpy as np

from src.benchmark.answering import _numbers, grade_correct
from src.benchmark.corpus import norm
from src.benchmark.metrics import bootstrap_mean, hop_ranks, paired_diff, question_scores
from src.optimization.config import OptimizationConfig
from src.optimization.token_budget import BudgetTier, TokenBudgetManager
from src.retrieval.labels import source_label


def test_hop_ranks_and_scores():
    ranked = [9, 4, 7, 1]
    hops = [{7}, {1, 2}]
    assert hop_ranks(ranked, hops) == [3, 4]
    s = question_scores(ranked, hops, ks=(3, 4))
    assert s["hop_recall@3"] == 0.5 and s["all_hops@3"] == 0.0
    assert s["hop_recall@4"] == 1.0 and s["all_hops@4"] == 1.0
    assert abs(s["rr"] - 1 / 3) < 1e-9


def test_missing_gold_scores_zero():
    s = question_scores([1, 2, 3], [{99}])
    assert s["rr"] == 0.0 and s["all_hops@8"] == 0.0


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    x = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], float)
    m, lo, hi = bootstrap_mean(x)
    assert lo <= m <= hi and (m, lo, hi) == bootstrap_mean(x)
    d, dlo, dhi = paired_diff(x, x)
    assert d == 0 and dlo == 0 and dhi == 0


def test_source_label_distinguishes_filings():
    a = source_label({"source_path": "data/raw\\sec-edgar-filings\\NVDA\\10-K\\0001045810-26-000021\\full-submission.txt"})
    b = source_label({"source_path": "data/raw_p3/sec-edgar-filings/AMD/10-K/0000002488-26-000018/full-submission.txt"})
    assert a == "NVDA 10-K 0001045810-26-000021" and b == "AMD 10-K 0000002488-26-000018"
    assert source_label({"source": "x/y/report.pdf"}) == "report.pdf"


def test_forced_tier_overrides_usage():
    bm = TokenBudgetManager(session_budget=1000)
    assert bm.get_tier() == BudgetTier.GREEN
    bm.forced_tier = BudgetTier.RED
    assert bm.get_tier() == BudgetTier.RED and not bm.should_use_heavy_model()
    assert bm.get_retrieval_top_n() == 4 and bm.should_skip_planner()


def test_bool_env_override(monkeypatch):
    monkeypatch.setenv("FINSIGHT_OPT_USE_CONTEXT_WINDOW", "false")
    assert OptimizationConfig().use_context_window is False
    monkeypatch.setenv("FINSIGHT_OPT_USE_CONTEXT_WINDOW", "true")
    assert OptimizationConfig().use_context_window is True


def test_numeric_match_shortcut_and_norm():
    assert _numbers("$26,974 million and 12.5%") == ["26974", "12.5"]
    assert grade_correct("q", "$26.97 billion", "Revenue was $26.97 billion [Doc 1]")["by"] == "numeric_match"
    assert norm("Apple’s  Net\nSales") == "apple's net sales"
