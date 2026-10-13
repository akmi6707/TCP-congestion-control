#!/usr/bin/env python3
"""Analyze persisted experiment results for one suite and print a summary.

Writes results/analysis/<suite>_<experiment>.json plus a markdown table at
results/analysis/summary_<suite>.md.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.evaluation.analyze import (analyze_experiment, latest_experiment,
                                    write_analysis)
from src.utils import io as uio

log = logging.getLogger("analyze_cli")

BASELINES = ["reno", "cubic", "bbr_like", "vegas_like"]
LEARNING = ["random", "ml_only", "rl_only", "hybrid",
            "rl_only_nodelay", "rl_only_noloss"]


def summary_markdown(analysis) -> str:
    lines = []
    am = analysis["aggregate"]
    ctrls = [c for c in BASELINES + LEARNING if c in am]
    lines.append(f"# Results summary: {analysis['experiment']}\n")
    lines.append(f"runs ok: {analysis['n_ok_runs']}, failed: {analysis['n_failed_runs']}\n")
    lines.append("| controller | goodput (Mbps) | mean RTT (ms) | p95 RTT (ms) | loss (%) | Jain |")
    lines.append("|---|---|---|---|---|---|")
    for c in ctrls:
        a = am.get(c, {})
        gp = a.get("m_goodput_mbps")
        mr = a.get("m_mean_rtt_ms")
        pr = a.get("m_p95_rtt_ms")
        ls = a.get("m_loss_rate")
        jn = a.get("m_jain_fairness")
        lines.append(
            f"| {c} | {gp:.2f} | {mr:.1f} | {pr:.1f} | "
            f"{100*ls:.2f} | {jn:.3f} |"
            if None not in (gp, mr, pr, ls, jn) else f"| {c} | ? | ? | ? | ? | ? |")
    tests = analysis.get("primary_tests", {})
    if tests:
        lines.append("\n## Paired tests (primary controller vs others, "
                     "goodput, one-sided Wilcoxon over (scenario, seed) pairs)\n")
        lines.append("| comparison | n | mean A | mean B | diff | p (raw) | Holm reject | Cliff's delta | rank-biserial | geo-ratio [CI] |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, t in tests.items():
            if "wilcoxon_p" not in t:
                continue
            geo = t.get("geo", {})
            ci = f"{geo.get('geo_mean_ratio')} [{geo.get('ci95_lo')}, {geo.get('ci95_hi')}]"
            lines.append(
                f"| {name} | {t['n_pairs']} | {t['mean_a']:.2f} | {t['mean_b']:.2f} | "
                f"{t['mean_diff']:+.2f} | {t['wilcoxon_p']:.4f} | "
                f"{t.get('holm_reject', '')} | {t.get('cliffs_delta'):+.2f} | "
                f"{t.get('rank_biserial'):+.2f} | {ci} |")
    guard = analysis.get("guardrails", {})
    if guard:
        lines.append("\n## Guardrails (aggregate means; ok=False = violated)\n")
        for bl, g in guard.items():
            for m, vals in g.items():
                if vals.get("ok") is None:
                    continue
                lines.append(f"- vs {bl}: {m} = {vals['a']:.3f} vs {vals['b']:.3f} "
                             f"-> {'OK' if vals['ok'] else 'VIOLATION'}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--primary", default="hybrid")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    exp = args.experiment or latest_experiment(Path(args.results_dir), args.suite)
    if exp is None:
        log.error("no experiment found for suite %s", args.suite)
        return 1
    analysis = analyze_experiment(Path(args.results_dir), exp,
                                  primary_controller=args.primary,
                                  baselines=BASELINES, learning=LEARNING)
    out_json = write_analysis(Path(args.results_dir), analysis,
                              f"{args.suite}_{exp}")
    out_md = Path(args.results_dir) / "analysis" / f"summary_{args.suite}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(summary_markdown(analysis), encoding="utf-8")
    print(summary_markdown(analysis))
    log.info("wrote %s and %s", out_json, out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
