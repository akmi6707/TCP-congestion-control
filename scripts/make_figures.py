#!/usr/bin/env python3
"""Generate publication-quality figures from persisted run records.

Reads only from results/ (never re-runs experiments) and writes PNGs to
figures/. Uses the most recent experiment for each suite unless overridden.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import io as uio
from src.evaluation.analyze import latest_experiment

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

CTRLS = ["reno", "cubic", "bbr_like", "vegas_like", "random",
         "ml_only", "rl_only", "hybrid", "rl_only_nodelay", "rl_only_noloss"]
COLORS = {
    "reno": "#d62728", "cubic": "#ff7f0e", "bbr_like": "#2ca02c",
    "vegas_like": "#8c564b", "random": "#bbbbbb", "ml_only": "#e377c2",
    "rl_only": "#1f77b4", "hybrid": "#9467bd",
    "rl_only_nodelay": "#17becf", "rl_only_noloss": "#bcbd22",
}
LABELS = {
    "reno": "Reno", "cubic": "CUBIC", "bbr_like": "BBR-like",
    "vegas_like": "Vegas-like", "random": "Random", "ml_only": "ML-only",
    "rl_only": "RL-only", "hybrid": "Hybrid (ML+RL)",
    "rl_only_nodelay": "RL (no-delay abl.)", "rl_only_noloss": "RL (no-loss abl.)",
}


def load_df(results_dir, suite):
    exp = latest_experiment(Path(results_dir), suite)
    if exp is None:
        return None, None
    records = uio.load_run_records(Path(results_dir), experiment=exp, status="ok")
    if not records:
        return exp, None
    df = uio.records_to_dataframe(records)
    # force numeric dtypes on the metric columns used for aggregation
    for col in ("m_goodput_mbps", "m_mean_rtt_ms", "m_p95_rtt_ms",
                "m_loss_rate", "m_jain_fairness", "m_mean_inference_ms",
                "m_p95_qd_ms", "m_mean_qd_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return exp, df


def fig_goodput_by_scenario(df, suite, out):
    ctrls = [c for c in CTRLS if c in df["controller"].unique()]
    scenarios = sorted(df["scenario"].unique())
    n_s, n_c = len(scenarios), len(ctrls)
    fig, ax = plt.subplots(figsize=(max(8, 0.62 * n_s * n_c ** 0.6), 3.4))
    width = 0.8 / n_c
    for i, c in enumerate(ctrls):
        g = df[df["controller"] == c].groupby("scenario")["m_goodput_mbps"].mean()
        sems = df[df["controller"] == c].groupby("scenario")["m_goodput_mbps"].sem()
        means = np.array([g.get(s, np.nan) for s in scenarios])
        errs = np.array([1.96 * (sems.get(s, np.nan) if np.isfinite(sems.get(s, np.nan) if s in sems.index else np.nan) else 0.0)
                         for s in scenarios])
        x = np.arange(n_s) + (i - n_c / 2) * width
        ax.bar(x, means, width * 0.92, yerr=errs, capsize=1.5,
               color=COLORS[c], label=LABELS[c], edgecolor="white", linewidth=0.3)
    ax.set_xticks(np.arange(n_s))
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("goodput (Mbps)")
    ax.set_title(f"Goodput by scenario ({suite} suite, mean over seeds, 95% CI)")
    ax.legend(ncol=min(n_c, 5), fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_pareto_rtt(df, suite, out):
    ctrls = [c for c in CTRLS if c in df["controller"].unique()]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for c in ctrls:
        g = df[df["controller"] == c].groupby("scenario")[["m_p95_rtt_ms", "m_goodput_mbps"]].mean()
        ax.scatter(g["m_p95_rtt_ms"], g["m_goodput_mbps"], s=22,
                   color=COLORS[c], label=LABELS[c], alpha=0.85,
                   edgecolor="black", linewidth=0.3)
    ax.set_xlabel("p95 RTT (ms), scenario mean")
    ax.set_ylabel("goodput (Mbps), scenario mean")
    ax.set_title(f"Throughput-latency trade-off ({suite} suite)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_loss_tradeoff(df, suite, out):
    ctrls = [c for c in CTRLS if c in df["controller"].unique()]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for c in ctrls:
        g = df[df["controller"] == c].groupby("scenario")[["m_loss_rate", "m_goodput_mbps"]].mean()
        ax.scatter(g["m_loss_rate"] * 100, g["m_goodput_mbps"], s=22,
                   color=COLORS[c], label=LABELS[c], alpha=0.85,
                   edgecolor="black", linewidth=0.3)
    ax.set_xlabel("loss rate (%), scenario mean")
    ax.set_ylabel("goodput (Mbps)")
    ax.set_title(f"Goodput vs loss ({suite} suite)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_training_curves(results_dir, out):
    logs = []
    for v in ["rl_only", "hybrid", "rl_only_nodelay", "rl_only_noloss"]:
        p = Path(results_dir) / "training" / f"dqn_{v}_log.json"
        if p.is_file():
            logs.append((v, json.loads(p.read_text())))
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for v, log in logs:
        vals = [(e["ep"], e["val_goodput_mbps"]) for e in log["episodes"]
                if "val_goodput_mbps" in e]
        if vals:
            x, y = zip(*vals)
            ax.plot(x, y, marker="o", ms=3, color=COLORS[v], label=LABELS[v])
            if "best_val_ma3" in log:
                ax.axhline(log["best_val_ma3"], color=COLORS[v], ls=":",
                           lw=0.8, alpha=0.7)
    ax.set_xlabel("training episode")
    ax.set_ylabel("validation mean goodput (Mbps)")
    ax.set_title("DQN training: validation goodput (dotted = best MA3)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_robustness_timeseries(results_dir, suite, out):
    exp = latest_experiment(Path(results_dir), suite)
    if exp is None:
        return
    records = uio.load_run_records(Path(results_dir), experiment=exp, status="ok")
    # plot the bw-down scenario if present, one line per controller
    by_ctrl = {}
    for r in records:
        sp = r.get("series_path")
        if not sp or r["scenario"]["name"] != "rob_bwdown_20to10":
            continue
        by_ctrl.setdefault(r["controller"], []).append(json.loads(Path(sp).read_text()))
    if not by_ctrl:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for c, series_list in sorted(by_ctrl.items()):
        s = series_list[0]
        t = np.array(s["t_s"]) / 1000.0
        gp = np.array(s["goodput_mbps"])
        cap = np.array(s["cap_mbps_mean"])
        ax.plot(t, gp, color=COLORS.get(c, "k"), label=LABELS.get(c, c), lw=1.2)
        if c == list(by_ctrl)[0]:
            ax.plot(t, cap, color="black", ls="--", lw=1.0, label="link capacity")
    ax.axvline(8.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("goodput (Mbps, 100 ms buckets)")
    ax.set_title("Robustness: capacity step 20->10 Mbps at t=8 s")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_window_dynamics(results_dir, suite, out):
    exp = latest_experiment(Path(results_dir), suite)
    if exp is None:
        return
    records = uio.load_run_records(Path(results_dir), experiment=exp, status="ok")
    want = {"hybrid", "reno", "bbr_like", "vegas_like"}
    target_sc = None
    by_ctrl = {}
    for r in records:
        sp = r.get("series_path")
        if not sp or r["controller"] not in want:
            continue
        if target_sc is None or r["scenario"]["name"] == "bw20_rtt60_q2_f1":
            target_sc = r["scenario"]["name"]
        if r["scenario"]["name"] == target_sc:
            by_ctrl.setdefault(r["controller"], []).append(
                json.loads(Path(sp).read_text()))
    if not by_ctrl:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    for c, series_list in sorted(by_ctrl.items()):
        s = series_list[0]
        t = np.array(s["t_s"]) / 1000.0
        cw = np.array(s["per_flow"][0]["cwnd_mean"])
        qd = np.array(s["per_flow"][0]["qd_mean"])
        ax1.plot(t, cw, color=COLORS.get(c, "k"), label=LABELS.get(c, c), lw=1.1)
        ax2.plot(t, qd, color=COLORS.get(c, "k"), lw=1.1)
    ax1.set_ylabel("cwnd (packets)")
    ax1.set_title(f"Window & queueing-delay dynamics ({target_sc})")
    ax1.legend(fontsize=7)
    ax2.set_ylabel("queueing delay (ms)")
    ax2.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_fairness(df, suite, out):
    rows = df[df["scenario"].str.contains("f4")]
    if rows.empty:
        return
    ctrls = [c for c in CTRLS if c in rows["controller"].unique()]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    means = [rows[rows["controller"] == c]["m_jain_fairness"].mean() for c in ctrls]
    ax.bar(range(len(ctrls)), means, color=[COLORS[c] for c in ctrls])
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(len(ctrls)))
    ax.set_xticklabels([LABELS[c] for c in ctrls], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Jain fairness index")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fairness with 4 competing flows")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_inference_cost(df, out):
    costs = df.groupby("controller")["m_mean_inference_ms"].mean() if "m_mean_inference_ms" in df else None
    if costs is None or costs.dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    cts = costs.dropna()
    ax.bar(range(len(cts)), cts.values,
           color=[COLORS.get(c, "k") for c in cts.index])
    ax.set_xticks(range(len(cts)))
    ax.set_xticklabels([LABELS.get(c, c) for c in cts.index], rotation=30,
                       ha="right", fontsize=7)
    ax.set_ylabel("mean inference latency (ms / control step)")
    ax.set_title("Controller inference cost")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True, parents=True)

    fig_training_curves(args.results_dir, out / "training_curves.png")

    for suite in ["core", "generalization", "robustness"]:
        exp, df = load_df(args.results_dir, suite)
        if df is None:
            logging.info("no data for suite %s", suite)
            continue
        logging.info("suite %s: experiment %s, %d runs", suite, exp, len(df))
        fig_goodput_by_scenario(df, suite, out / f"goodput_{suite}.png")
        fig_pareto_rtt(df, suite, out / f"pareto_{suite}.png")
        fig_loss_tradeoff(df, suite, out / f"loss_tradeoff_{suite}.png")
        fig_fairness(df, suite, out / f"fairness_{suite}.png")
    fig_robustness_timeseries(args.results_dir, "robustness",
                              out / "robustness_timeseries.png")
    fig_window_dynamics(args.results_dir, "core", out / "window_dynamics.png")
    logging.info("figures written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
