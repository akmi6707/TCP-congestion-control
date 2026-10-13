"""Statistical analysis of experiment results.

Pre-registered procedure (configs/evaluation_protocol.yaml):
- primary metric: goodput_mbps; confirmatory comparisons: hybrid vs each
  conventional baseline on the core suite, paired across (scenario, seed)
  runs (same scenario + seed => same link-noise sequence).
- one-sided paired Wilcoxon signed-rank on the goodput differences,
  Holm-corrected across the baseline family (alpha=0.05).
- effect sizes: Cliff's delta and matched-pairs rank-biserial correlation.
- aggregate summary: geometric mean of per-scenario goodput ratios with a
  cluster (scenario-level) bootstrap 95% CI.
- guardrails evaluated on aggregate means (mean/p95 RTT, loss, Jain).

All p-values and effect sizes are computed from actually measured run
records; nothing is imputed. Runs with status != ok are excluded and
reported as failures.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..utils import io as uio

PRIMARY = "m_goodput_mbps"
GUARDRAILS = ["m_mean_rtt_ms", "m_p95_rtt_ms", "m_loss_rate", "m_jain_fairness"]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta between paired samples (x_i preferred over y_i)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gt = lt = eq = 0
    for a, b in zip(x, y):
        if a > b:
            gt += 1
        elif a < b:
            lt += 1
        else:
            eq += 1
    n = len(x)
    return (gt - lt) / n if n else float("nan")


def rank_biserial_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby simple method)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    sum_pos = ranks[d > 0].sum()
    sum_neg = ranks[d < 0].sum()
    return float((sum_pos - sum_neg) / ranks.sum())


def holm_correction(pvals: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni: returns reject flags."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rejected = [False] * m
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if pvals[i] <= threshold and all(rejected[j] for j in order[:rank]):
            rejected[i] = True
        else:
            break
    return rejected


def paired_test(df_a, df_b, metric: str, alternative: str = "greater") -> Dict:
    """Paired comparison df_a vs df_b on metric, joined on (scenario, seed)."""
    a = df_a[["scenario", "seed", metric]].rename(columns={metric: "a"})
    b = df_b[["scenario", "seed", metric]].rename(columns={metric: "b"})
    merged = a.merge(b, on=["scenario", "seed"], how="inner").dropna()
    x = merged["a"].to_numpy(dtype=float)
    y = merged["b"].to_numpy(dtype=float)
    n = len(x)
    out = {"n_pairs": n, "mean_a": float(x.mean()) if n else None,
           "mean_b": float(y.mean()) if n else None,
           "mean_diff": float((x - y).mean()) if n else None}
    if n >= 3:
        zero_method = "wilcox" if np.any(x != y) else "pratt"
        try:
            w = stats.wilcoxon(x, y, alternative=alternative, zero_method=zero_method)
            out["wilcoxon_p"] = float(w.pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
        out["cliffs_delta"] = cliffs_delta(x, y)
        out["rank_biserial"] = rank_biserial_wilcoxon(x, y)
        if n >= 5:
            t = stats.ttest_rel(x, y, alternative=alternative)
            out["paired_t_p"] = float(t.pvalue)
    return out


def geometric_mean_ratios(df_a, df_b, metric: str, n_boot: int = 10000,
                          rng_seed: int = 0) -> Dict:
    """Geometric mean over scenarios of per-scenario mean a/b ratios, with a
    scenario-level (cluster) bootstrap 95% CI."""
    a = df_a.groupby("scenario")[metric].mean()
    b = df_b.groupby("scenario")[metric].mean()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    ratios = (a / b.replace(0, np.nan)).dropna().to_numpy(dtype=float)
    if len(ratios) == 0:
        return {"geo_mean_ratio": None}
    g = float(np.exp(np.mean(np.log(ratios))))

    def gm(rs):
        return float(np.exp(np.mean(np.log(rs))))

    rng = np.random.default_rng(rng_seed)
    boots = np.array([
        gm(ratios[rng.integers(0, len(ratios), len(ratios))])
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"geo_mean_ratio": g, "ci95_lo": float(lo), "ci95_hi": float(hi),
            "n_scenarios": int(len(ratios))}


def aggregate_means(df, metrics: List[str]) -> Dict[str, Optional[float]]:
    out = {}
    for m in metrics:
        vals = df.groupby("scenario")[m].mean().dropna()
        out[m] = float(vals.mean()) if len(vals) else None
    return out


def analyze_experiment(results_dir: Path, experiment: str,
                       primary_controller: str = "hybrid",
                       baselines: Optional[List[str]] = None,
                       learning: Optional[List[str]] = None,
                       guardrail_defs: Optional[Dict] = None) -> Dict:
    records = uio.load_run_records(results_dir, experiment=experiment, status="ok")
    failed = uio.load_run_records(results_dir, experiment=experiment, status="failed")
    if not records:
        raise RuntimeError(f"no ok records for experiment {experiment}")
    df = uio.records_to_dataframe(records)
    controllers = sorted(df["controller"].unique())
    baselines = [b for b in (baselines or ["reno", "cubic", "bbr_like", "vegas_like"])
                 if b in controllers]
    learning = [c for c in (learning or ["random", "ml_only", "rl_only", "hybrid",
                                         "rl_only_nodelay", "rl_only_noloss"])
                if c in controllers]

    # ---------------- per-scenario x controller table ------------------
    table = {}
    for (sc, ctrl), g in df.groupby(["scenario", "controller"]):
        row = {"n": int(len(g)), "seed_values": sorted(g["seed"].tolist())}
        for m in [PRIMARY] + GUARDRAILS:
            v = g[m].dropna().to_numpy(dtype=float)
            if len(v):
                row[m] = float(v.mean())
                row[m + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                # 95% t-CI on the per-seed mean
                if len(v) > 1:
                    ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
                    row[m + "_ci95"] = float(ci)
                else:
                    row[m + "_ci95"] = 0.0
            else:
                row[m] = None
        table[f"{sc}|{ctrl}"] = row

    # ---------------- primary hypothesis tests -------------------------
    pc = primary_controller
    if pc in controllers:
        df_pc = df[df["controller"] == pc]
        tests = {}
        pvals = []
        for b in baselines:
            df_b = df[df["controller"] == b]
            t = paired_test(df_pc, df_b, PRIMARY, alternative="greater")
            t["geo"] = geometric_mean_ratios(df_pc, df_b, PRIMARY)
            tests[b] = t
            pvals.append(t.get("wilcoxon_p", float("nan")))
        # Holm across the baseline family
        valid = [i for i, p in enumerate(pvals) if not math.isnan(p)]
        corrected = holm_correction([pvals[i] for i in valid], alpha=0.05)
        for k, i in enumerate(valid):
            tests[baselines[i]]["holm_reject"] = bool(corrected[k])
        # learning-family comparisons (descriptive + RQ2)
        for c in learning:
            if c == pc:
                continue
            df_c = df[df["controller"] == c]
            tests[c] = paired_test(df_pc, df_c, PRIMARY, alternative="greater")
            tests[c]["geo"] = geometric_mean_ratios(df_pc, df_c, PRIMARY)
        # RQ2 latency side: p95 rtt, hybrid better = smaller
        if "rl_only" in controllers:
            tests["rl_only_p95rtt"] = paired_test(df_pc, df[df["controller"] == "rl_only"],
                                                  "m_p95_rtt_ms", alternative="less")
    else:
        tests = {}

    # ---------------- guardrails --------------------------------------
    guard = {}
    if pc in controllers and baselines:
        agg_pc = aggregate_means(df[df["controller"] == pc], [PRIMARY] + GUARDRAILS)
        for b in baselines:
            agg_b = aggregate_means(df[df["controller"] == b], [PRIMARY] + GUARDRAILS)
            g = {}
            for m in GUARDRAILS:
                va, vb = agg_pc.get(m), agg_b.get(m)
                if va is None or vb is None:
                    g[m] = {"a": va, "b": vb, "ok": None}
                    continue
                if m == "m_jain_fairness":
                    ok = va >= vb - 0.02
                elif m == "m_loss_rate":
                    ok = (va <= vb + 0.005) or (va <= 2.0 * max(vb, 1e-9))
                else:
                    ok = va <= 1.10 * max(vb, 1e-9)
                g[m] = {"a": va, "b": vb, "ok": bool(ok)}
            guard[b] = g

    # ---------------- inference cost ----------------------------------
    costs = {}
    for c in controllers:
        rows = df[df["controller"] == c]
        inf = [r.get("inference") or {} for r in records
               if r.get("controller") == c]
        ms = [i["mean_ms"] for i in inf if i.get("mean_ms") is not None]
        if ms:
            costs[c] = {"mean_inference_ms": float(np.mean(ms)),
                        "max_inference_ms": float(np.max([i.get("max_ms", 0) or 0 for i in inf]))}

    return {
        "experiment": experiment,
        "n_ok_runs": len(records),
        "n_failed_runs": len(failed),
        "failed_reasons": [f.get("error") for f in failed[:10]],
        "controllers": controllers,
        "scenarios": sorted(df["scenario"].unique().tolist()),
        "per_run_table": table,
        "primary_tests": tests,
        "guardrails": guard,
        "aggregate": {c: aggregate_means(df[df["controller"] == c],
                                         [PRIMARY] + GUARDRAILS)
                      for c in controllers},
        "inference_cost": costs,
    }


def latest_experiment(results_dir: Path, suite: str) -> Optional[str]:
    """Most recent experiment id for a suite (by run record timestamps)."""
    records = uio.load_run_records(results_dir, status=None)
    exps = {}
    for r in records:
        if r.get("suite") == suite:
            ts = r.get("timestamp_utc", "") or r.get("run_id", "")
            exps[r.get("experiment")] = max(exps.get(r.get("experiment"), ""), ts)
    if not exps:
        return None
    return max(exps.items(), key=lambda kv: kv[1])[0]


def write_analysis(results_dir: Path, analysis: Dict, out_name: str) -> Path:
    out = Path(results_dir) / "analysis" / f"{out_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=1, default=float)
    return out
