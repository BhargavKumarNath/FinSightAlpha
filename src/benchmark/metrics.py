"""Retrieval metrics and bootstrap utilities for the Phase 3 evaluation."""

from __future__ import annotations

import numpy as np


def hop_ranks(ranked_ids: list[int], gold_hops: list[set[int]]) -> list[int | None]:
    """1-based rank of the first gold chunk of each hop, None if absent."""
    out: list[int | None] = []
    for hop in gold_hops:
        r = next((i + 1 for i, cid in enumerate(ranked_ids) if cid in hop), None)
        out.append(r)
    return out


def question_scores(ranked_ids: list[int], gold_hops: list[set[int]], ks=(1, 3, 5, 8)) -> dict:
    """Per-question retrieval scores.

    hop_recall@k: fraction of hops with a gold chunk in the top-k.
    all_hops@k: 1 if every hop is covered (strict; equals hop_recall for single-hop).
    rr: reciprocal rank of the first gold chunk of any hop, 0 if none retrieved.
    """
    ranks = hop_ranks(ranked_ids, gold_hops)
    res: dict[str, float] = {}
    for k in ks:
        cov = [r is not None and r <= k for r in ranks]
        res[f"hop_recall@{k}"] = float(np.mean(cov))
        res[f"all_hops@{k}"] = float(all(cov))
    found = [r for r in ranks if r is not None]
    res["rr"] = 1.0 / min(found) if found else 0.0
    return res


def bootstrap_mean(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile CI, resampling questions."""
    x = np.asarray(x, dtype=float)
    assert x.ndim == 1 and len(x) > 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a-b) with a 95% paired bootstrap CI over questions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1
    return bootstrap_mean(a - b, n, seed)
