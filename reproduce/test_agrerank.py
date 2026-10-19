"""Unit tests for the equations in agrerank.py.  Run: python -m pytest reproduce -q"""
import numpy as np

import agrerank as ag


def _pool(n=50, d=16, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n, d))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    q = rng.normal(size=d)
    q /= np.linalg.norm(q)
    return q, D, rng.gamma(2.0, 3.0, size=n)


def test_percentile_eq1():
    assert np.allclose(ag.percentile(np.array([12.1, 3.0, 8.7, 25.4, 9.9])), [0.75, 0, 0.25, 1, 0.5])
    assert np.allclose(ag.percentile(np.array([1.0, 1.0, 2.0])), [0.25, 0.25, 1.0])  # ties: average rank


def test_rrf_matches_definition():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])
    assert np.allclose(ag.rrf(a, b), [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62])


def test_seeds_are_consensus_capped_and_weighted():
    q, D, sp = _pool(200)
    tr = ag.Trace()
    ag.agrerank(q, D, sp, None, ag.Config(stop_after="expand"), trace=tr)
    ps, pd = ag.percentile(sp), ag.percentile(D @ q)
    cons = set(np.flatnonzero((ps > 0.8) & (pd > 0.8)))
    assert tr.n_consensus == len(cons)
    if not tr.fallback:
        assert set(tr.seeds) <= cons and len(tr.seeds) == min(15, len(cons))
        w = ps[tr.seeds] * pd[tr.seeds]
        assert np.all(np.diff(w) <= 1e-12)  # strongest consensus first


def test_fallback_uses_rrf_top10():
    q, D, _ = _pool(100)
    sp = -(D @ q)  # sparse perfectly anti-correlated with dense: no consensus
    tr = ag.Trace()
    ag.agrerank(q, D, sp, None, ag.Config(stop_after="expand"), trace=tr)
    assert tr.fallback and len(tr.seeds) == 10
    assert set(tr.seeds) == set(ag.top_m(ag.rrf(sp, D @ q), 10))


def test_expansion_eq3():
    q, D, sp = _pool(200)
    tr = ag.Trace()
    s = ag.agrerank(q, D, sp, None, ag.Config(stop_after="expand"), trace=tr)
    ps, pd = ag.percentile(sp), ag.percentile(D @ q)
    w = ps[tr.seeds] * pd[tr.seeds] if not tr.fallback else np.ones(len(tr.seeds))
    qp = 0.65 * q + 0.35 * (w[:, None] * D[tr.seeds]).sum(0) / w.sum()
    qp /= np.linalg.norm(qp)
    assert np.allclose(s, D @ qp)


def test_alpha_one_is_dense_order():
    q, D, sp = _pool()
    s = ag.agrerank(q, D, sp, None, ag.Config(alpha=1.0, stop_after="expand"))
    assert np.allclose(s, D @ q)


def test_gate_eq4_eq5():
    q, D, sp = _pool()
    s_exp = ag.agrerank(q, D, sp, None, ag.Config(stop_after="expand"))
    s3 = ag.agrerank(q, D, sp, None, ag.Config(stop_after="gate"))
    g = 0.5 + 0.5 * np.maximum(0, D @ q)
    assert np.allclose(s3, s_exp * g) and g.min() >= 0.5 and g.max() <= 1.0


def test_final_fusion_eq6_and_depth():
    q, D, sp = _pool(300)
    logits = np.random.default_rng(1).normal(size=300)
    calls = []
    ce = lambda idx: (calls.append(len(idx)), logits[idx])[1]
    cfg = ag.Config(M=40)
    s = ag.agrerank(q, D, sp, ce, cfg)
    s3 = ag.agrerank(q, D, sp, None, ag.Config(stop_after="gate"))
    top = ag.top_m(s3, 40)
    assert calls == [40]
    expect = 0.5 * ag.minmax(logits[top]) + 0.5 * ag.percentile(s3[top])
    assert np.allclose(s[top], expect)
    rest = np.setdiff1d(np.arange(300), top)
    assert s[rest].max() < s[top].min()  # nothing outside the CE pool outranks it


def test_pure_ce_baseline_orders_by_ce():
    rng = np.random.default_rng(2)
    first, logits = rng.normal(size=100), rng.normal(size=100)
    s = ag.rerank_top(first, lambda idx: logits[idx], 20)
    top = ag.top_m(first, 20)
    assert list(ag.top_m(s, 20)) == list(top[np.argsort(-logits[top])])


def test_requery_adds_documents():
    q, E, sp = _pool(400)
    pool = ag.top_m(sp, 50)
    union, s = ag.agrerank_requery(q, E, sp, pool, None, ag.Config(), k_requery=50)
    assert set(pool) <= set(union) and len(union) > len(pool) and len(s) == len(union)
