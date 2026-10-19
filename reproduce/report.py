"""Write REPRODUCTION.md: the paper's numbers next to the reproduced ones."""
from __future__ import annotations

import json

from infra import ROOT

OUT = ROOT / "outputs"

PAPER_T2 = {  # Table 2, k=1000: (nDCG@10, MAP, MRR)
    "2021": {"BM25": (.254, .132, .567), "Dense": (.666, .394, .962), "RRF": (.518, .307, .891),
             "BM25+CE": (.576, .223, .937), "Dense+CE": (.683, .406, .959), "RRF+CE": (.681, .396, .959),
             "AgreRank": (.706, .426, .981)},
    "2022": {"BM25": (.132, .045, .378), "Dense": (.551, .179, .898), "RRF": (.373, .126, .765),
             "BM25+CE": (.404, .085, .768), "Dense+CE": (.593, .186, .908), "RRF+CE": (.594, .183, .912),
             "AgreRank": (.627, .198, .941)},
}
PAPER_DEPTH = {200: (.522, .551), 500: (.563, .602), 1000: (.594, .627)}  # DL'22 (RRF+CE, AgreRank)
PAPER_ABL = [("RRF", .363), ("abl: +consensus seeding", .257), ("abl: +anchor expansion", .532),
             ("abl: +geometric gating", .536), ("AgreRank", .602), ("abl: RRF seeds", .590), ("abl: no gating", .595)]


def pval(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    T = json.loads((OUT / "trec_results.json").read_text())
    L = []
    w = L.append
    w("# Reproduction report\n")
    w("Every number below was recomputed from scratch by `reproduce/run_all.sh` "
      "(see [`reproduce/README.md`](reproduce/README.md)). Raw results: `reproduce/outputs/`.\n")
    a21, a22 = (T[f"dl{y}_k1000"]["systems"]["AgreRank"]["ndcg10"] for y in ("2021", "2022"))
    w(f"**Summary.** AgreRank reaches nDCG@10 {a21:.3f} on TREC DL 2021 and {a22:.3f} on DL 2022 "
      f"(paper: 0.706 and 0.627), and beats the Dense+CE and RRF+CE baselines on both years and at every "
      f"retrieval depth, as in the paper.\n")

    w("## TREC DL, k = 1000 (Table 2)\n")
    for year in ("2021", "2022"):
        S = T[f"dl{year}_k1000"]["systems"]
        w(f"**DL {year}** ({T[f'dl{year}_k1000']['n_queries']} queries)\n")
        w("| System | nDCG@10 paper | reproduced | MAP paper | reproduced | MRR paper | reproduced |")
        w("|---|---|---|---|---|---|---|")
        for name, (n, m, r) in PAPER_T2[year].items():
            x = S[name]
            w(f"| {name} | {n:.3f} | {x['ndcg10']:.3f} | {m:.3f} | {x['map']:.3f} | {r:.3f} | {x['mrr']:.3f} |")
        x = S["AgreRank (original code)"]
        w(f"| AgreRank, original code | | {x['ndcg10']:.3f} | | {x['map']:.3f} | | {x['mrr']:.3f} |")
        w("")
    w("BM25, RRF and BM25+CE come out higher than in the paper because the sparse scores here are "
      "Pyserini's BM25; the original script re-scored the pool with an in-pool `rank_bm25`. "
      "\"Original code\" is an exact port of `src/reranker_eval/agre_rank_with_baselines.py`.\n")

    w("## Significance (paired, per-query nDCG@10, k = 1000)\n")
    w("| Year | Comparison | Δ | 95% CI | t-test p | wins/ties/losses |")
    w("|---|---|---|---|---|---|")
    for y in ("2021", "2022"):
        for comp in ("AgreRank vs Dense+CE", "AgreRank vs RRF+CE"):
            t = T[f"dl{y}_k1000"]["tests_ndcg10"][comp]
            w(f"| DL {y} | {comp} | {t['diff']:+.3f} | [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {pval(t['t_p'])} "
              f"| {t['wins']}/{t['ties']}/{t['losses']} |")
    w("")

    w("## Retrieval depth, DL'22 (Figure 3)\n")
    w("| k | RRF+CE paper | reproduced | AgreRank paper | reproduced |")
    w("|---|---|---|---|---|")
    for k, (pb, pa) in PAPER_DEPTH.items():
        S = T[f"dl2022_k{k}"]["systems"]
        w(f"| {k} | {pb:.3f} | {S['RRF+CE']['ndcg10']:.3f} | {pa:.3f} | {S['AgreRank']['ndcg10']:.3f} |")
    w("\nDL'21: " + ", ".join(f"k={k}: {T[f'dl2021_k{k}']['systems']['AgreRank']['ndcg10']:.3f} vs RRF+CE "
                              f"{T[f'dl2021_k{k}']['systems']['RRF+CE']['ndcg10']:.3f}" for k in (200, 500, 1000)) + "\n")

    w("## Ablation, DL'22, k = 500 (Table 5)\n")
    S = T["dl2022_k500"]["systems"]
    w("| Configuration | paper | reproduced |")
    w("|---|---|---|")
    for name, p in PAPER_ABL:
        w(f"| {name.replace('abl: ', '')} | {p:.3f} | {S[name]['ndcg10']:.3f} |")
    w(f"| no expansion (alpha = 1) | 0.586 | {S['abl: no expansion']['ndcg10']:.3f} |")
    w("")

    w("## Sensitivity, DL'22, k = 500 (Figure 5, Table 6a)\n")
    for g in ("tau", "alpha", "beta", "M", "gamma"):
        ks = [k for k in S if k.startswith(f"sens {g}=")]
        w(f"- **{g}**: " + ", ".join(f"{k.split('=')[1]} → {S[k]['ndcg10']:.3f}" for k in ks))
    w("")

    w("## Notes\n")
    w("- **CE normalization.** Eq. 6 puts the cross-encoder score on [0, 1] by min–max over the top-M "
      "logits. A sigmoid saturates (top-10 probabilities all > 0.99 for most queries) and costs about two points.")
    w("- **Stage 2 on TREC.** All systems re-rank the same BM25 top-k pool; the expanded query re-scores that "
      "pool. On the diagnostic corpus, where exhaustive search is cheap, the full re-query is also reported.")
    s = T["dl2022_k1000"]["systems"]
    s1 = T["dl2021_k1000"]["systems"]
    w(f"- **Fusion control.** Giving Dense+CE the same CE + first-stage fusion as AgreRank yields "
      f"{s1['Dense+CE fused']['ndcg10']:.3f} (DL'21) and {s['Dense+CE fused']['ndcg10']:.3f} (DL'22), close to AgreRank; "
      f"the same fusion on RRF+CE yields {s1['RRF+CE fused']['ndcg10']:.3f} and {s['RRF+CE fused']['ndcg10']:.3f}.")
    sd = T["dl2022_k500"]["seeds"]
    w(f"- **Fallback.** With Pyserini BM25 scores, {sd['fallback_queries']} of 76 DL'22 queries need the RRF fallback "
      f"(mean {sd['mean_consensus_docs']:.0f} consensus documents per query); the paper reports 8.\n")

    dpath = OUT / "diag_results.json"
    if dpath.exists():
        Dg = json.loads(dpath.read_text())
        w("## Diagnostic corpus (100 queries, 1,249 → 11,249 passages)\n")
        c = Dg["clean"]["consensus_regions"]
        tot = sum(c["relevant"].values()) + c["relevant_outside_pool"]
        w("**Where relevant passages sit (Table 3, clean corpus)**\n")
        w("| Region | paper | reproduced |")
        w("|---|---|---|")
        paper3 = {"both": "141 / 145 (97%)", "dense_only": "1", "bm25_only": "2", "neither": "1"}
        for r in ("both", "dense_only", "bm25_only", "neither"):
            w(f"| {r.replace('_', ' ')} high | {paper3[r]} | {c['relevant'][r]} / {tot} ({c['relevant'][r] / tot:.0%}) |")
        w("")
        w("**Noise resilience (Table 4, MRR@10 on the 50 factoid queries, as in the paper's code)**\n")
        paper4 = {"BM25": (.487, .353), "Dense": (.417, .419), "RRF": (.513, .442), "RRF+CE": (.581, .523),
                  "AgreRank-requery": (.556, .546)}
        legacy_key = {"BM25": "legacy-BM25", "Dense": "Dense", "RRF": "legacy-RRF", "RRF+CE": "legacy-RRF+CE",
                      "AgreRank-requery": "legacy-AgreRank-pool"}
        w("| System | paper clean → noisy | reproduced, Pyserini BM25 | reproduced, paper's BM25 (rank_bm25) |")
        w("|---|---|---|---|")
        Sy = lambda c, k: Dg[c]["systems"][k]["factoid"]["mrr10"] if k in Dg[c]["systems"] else None
        for key, (pc, pn) in paper4.items():
            a = f"{Sy('clean', key):.3f} → {Sy('noisy', key):.3f}"
            lk = legacy_key[key]
            b = f"{Sy('clean', lk):.3f} → {Sy('noisy', lk):.3f}" if Sy("clean", lk) is not None else "–"
            w(f"| {key.replace('-requery', '')} | {pc:.3f} → {pn:.3f} | {a} | {b} |")
        w("\nAgreRank stays flat under noise and above RRF+CE, as in the paper. The BM25 drop reproduces exactly "
          "with the `rank_bm25` scorer the paper's code used; with Lucene BM25 the template distractors do not hurt it. "
          "RRF+CE does not degrade here (the paper's clean 0.581 is not reproduced). "
          "On this corpus the Stage-2 re-query retrieves nothing the initial pool lacks, so pool and re-query "
          "versions coincide.\n")
        w("**Expansion weight (Table 6b, clean corpus)**\n")
        w("| alpha | paper factoid MRR / synthesis recall | reproduced factoid MRR@10 / synthesis R@10 |")
        w("|---|---|---|")
        for a, (pm, pr) in {1.0: (.610, .750), 0.65: (.546, .938), 0.5: (.510, .938)}.items():
            R = Dg["clean"]["systems"]["AgreRank-requery" + ("" if a == 0.65 else f"-alpha={a}")]
            w(f"| {a} | {pm:.3f} / {pr:.3f} | {R['factoid']['mrr10']:.3f} / {R['synthesis']['r10']:.3f} |")
        w("\nThe paper's Table 6b was computed on an earlier 41-query version of the query set "
          "(`data/qset.json`); on the 100 queries, factoid MRR and synthesis recall barely move with alpha.\n")

    enc = {e: json.loads((OUT / f"trec_results_{e}.json").read_text())
           for e in ("bge", "e5") if (OUT / f"trec_results_{e}.json").exists()}
    if enc:
        w("## Other encoders, DL'22, k = 500 (Table 6c)\n")
        w("| Encoder | paper RRF+CE → AgreRank | reproduced RRF+CE → AgreRank |")
        w("|---|---|---|")
        paper6c = {"openai": ".563 → .602", "bge": ".505 → .579", "e5": ".521 → .582"}
        for e, R in {"openai": T, **enc}.items():
            S = R["dl2022_k500" + ("" if e == "openai" else f"_{e}")]["systems"]
            w(f"| {e} | {paper6c[e]} | {S['RRF+CE']['ndcg10']:.3f} → {S['AgreRank']['ndcg10']:.3f} |")
        w("\nAgreRank lands on the paper's values. The reproduced RRF+CE is stronger than the paper's because "
          "the original script for this table averaged the CE score with RRF instead of using the CE order, "
          "so the gains are smaller here but point the same way.\n")
    (ROOT.parent / "REPRODUCTION.md").write_text("\n".join(L) + "\n")
    print("wrote REPRODUCTION.md")


if __name__ == "__main__":
    main()
