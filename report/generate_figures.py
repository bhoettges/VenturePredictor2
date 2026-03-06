#!/usr/bin/env python3
"""
Generate figures for the Venture Prophet tech report.

Outputs PNG files into report/figures/.

This script is intentionally lightweight:
- No training runs
- Uses the production dataset (202402_Copy.csv)
- Uses stored model metadata from lightgbm_financial_model.pkl
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
DATA_PATH = ROOT / "202402_Copy.csv"
MODEL_PATH = ROOT / "lightgbm_financial_model.pkl"

# Ensure matplotlib/font caches are writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure repo-root imports (e.g., trend_detector.py) work when running from anywhere.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trend_detector import TrendDetector


def _savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def load_model_metadata() -> dict:
    with open(MODEL_PATH, "rb") as f:
        d = pickle.load(f)
    return d


def make_missingness_top30(df: pd.DataFrame) -> None:
    missing = (df.isna().mean() * 100).sort_values(ascending=False).head(30)
    plt.figure(figsize=(10, 7))
    missing.sort_values().plot(kind="barh", color="#375a7f")
    plt.xlabel("Missingness (%)")
    plt.title("Top 30 Features by Missingness (202402_Copy.csv)")
    _savefig("fig_01_missingness_top30.png")


def make_basic_distributions(df: pd.DataFrame) -> None:
    # cARR distribution (log10)
    if "cARR" in df.columns:
        x = pd.to_numeric(df["cARR"], errors="coerce").dropna()
        x = x.clip(lower=1)
        plt.figure(figsize=(9, 5))
        plt.hist(np.log10(x), bins=60, color="#2ca02c", alpha=0.85)
        plt.xlabel("log10(cARR)")
        plt.ylabel("Count (company-quarters)")
        plt.title("Distribution of ARR (log scale)")
        _savefig("fig_02_carr_log_distribution.png")

    # ARR YoY Growth distribution (clipped)
    yoy_col = "ARR YoY Growth (in %)"
    if yoy_col in df.columns:
        g = pd.to_numeric(df[yoy_col], errors="coerce").dropna()
        g = g.clip(-1.0, 5.0)  # readability: [-100%, +500%]
        plt.figure(figsize=(9, 5))
        plt.hist(g, bins=70, color="#ff7f0e", alpha=0.85)
        plt.xlabel("ARR YoY Growth (rate, clipped to [-1, 5])")
        plt.ylabel("Count (company-quarters)")
        plt.title("Distribution of ARR YoY Growth")
        _savefig("fig_03_yoy_growth_distribution.png")


def make_geography_and_sector(df: pd.DataFrame) -> None:
    def bar_top(col: str, n: int, title: str, filename: str) -> None:
        if col not in df.columns:
            return
        vc = df[col].fillna("Unknown").value_counts().head(n)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=vc.index.astype(str), y=vc.values, color="#4c72b0")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Count (company-quarters)")
        plt.title(title)
        _savefig(filename)

    bar_top("Country", 8, "Top Countries in Training Data", "fig_04_country_distribution.png")
    bar_top("Sector", 10, "Top Sectors in Training Data", "fig_05_sector_distribution.png")


def make_model_performance(model_meta: dict) -> None:
    perf = model_meta.get("performance_results", {})
    if not perf:
        return

    rows = []
    for target, metrics in perf.items():
        rows.append(
            {
                "horizon": target,
                "R2": float(metrics.get("R2", np.nan)),
                "MAE": float(metrics.get("MAE", np.nan)),
            }
        )
    perf_df = pd.DataFrame(rows).sort_values("horizon")

    # R2 per horizon
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=perf_df, x="horizon", y="R2", color="#2ca02c")
    plt.ylim(0, 1)
    plt.title("Model Performance by Forecast Horizon (R²)")
    _savefig("fig_06_r2_by_horizon.png")

    # MAE per horizon
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=perf_df, x="horizon", y="MAE", color="#d62728")
    plt.title("Model Performance by Forecast Horizon (MAE)")
    _savefig("fig_07_mae_by_horizon.png")


def make_feature_importance(model_meta: dict) -> None:
    fi = model_meta.get("feature_importance", None)
    cols = model_meta.get("feature_cols", None)
    if fi is None or cols is None:
        return

    # In this repo, `feature_importance` is typically a DataFrame with per-horizon
    # importances and `mean_importance`. The index corresponds to the feature order.
    if isinstance(fi, pd.DataFrame):
        imp_df = fi.copy()
        if len(imp_df) != len(cols):
            # If we can't align to feature_cols, fall back to legacy plots.
            return
        imp_df = imp_df.reset_index(drop=True)
        imp_df["feature"] = cols
        importance_col = "mean_importance" if "mean_importance" in imp_df.columns else imp_df.columns[-1]
        imp = imp_df[["feature", importance_col]].rename(columns={importance_col: "importance"})
    else:
        fi_arr = np.asarray(fi, dtype=float)
        if fi_arr.ndim != 1 or len(fi_arr) != len(cols):
            return
        imp = pd.DataFrame({"feature": cols, "importance": fi_arr})

    imp = imp.sort_values("importance", ascending=False).head(20).copy()

    plt.figure(figsize=(10, 7))
    sns.barplot(data=imp, y="feature", x="importance", color="#9467bd")
    plt.title("Top 20 Features by Model Importance (LightGBM)")
    _savefig("fig_08_feature_importance_top20.png")


def make_trend_routing_breakdown(df: pd.DataFrame) -> None:
    """
    Compute trend types on the *last 4 quarters per company* as a proxy for
    how often the hybrid router would trigger different regimes.
    """
    if "id_company" not in df.columns or "Financial Quarter" not in df.columns or "cARR" not in df.columns:
        return

    # Parse multiple quarter formats -> time index.
    # Dataset format: "Q1-FY19", "Q4-FY23", etc.
    # API/user format: "Q1 2023", "FY24 Q1" (handled defensively).
    tmp = df[["id_company", "Financial Quarter", "cARR"]].copy()
    fq = tmp["Financial Quarter"].astype(str)

    # Quarter number
    q = fq.str.extract(r"(Q[1-4])")[0]
    tmp["Quarter Num"] = q.map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})

    # Year: prefer FYxx patterns, otherwise 4-digit year.
    fy = fq.str.extract(r"FY(\d{2,4})")[0]
    y4 = fq.str.extract(r"(19\d{2}|20\d{2})")[0]
    year = fy.fillna(y4)
    tmp["Year"] = pd.to_numeric(year, errors="coerce")
    tmp = tmp.dropna(subset=["Year", "Quarter Num"])
    tmp["Year"] = tmp["Year"].apply(lambda x: int(x + 2000) if x < 100 else int(x))
    tmp["time_idx"] = tmp["Year"].astype(int) * 4 + tmp["Quarter Num"].astype(int)
    tmp = tmp.sort_values(["id_company", "time_idx"])

    detector = TrendDetector()

    trend_rows = []
    for company_id, g in tmp.groupby("id_company"):
        g = g.dropna(subset=["cARR"])
        if len(g) < 4:
            continue
        last4 = g.tail(4)["cARR"].astype(float).tolist()
        q1, q2, q3, q4 = last4
        if min(last4) <= 0:
            continue
        tr = detector.detect_trend(q1, q2, q3, q4)
        trend_rows.append({"trend_type": tr["trend_type"], "use_gpt_flag": bool(tr["use_gpt"])})

    if not trend_rows:
        return

    tr_df = pd.DataFrame(trend_rows)
    counts = tr_df["trend_type"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values, color="#1f77b4")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Companies (last-4Q regime)")
    plt.title("Hybrid Router Regimes (Trend Types on Last 4 Quarters per Company)")
    _savefig("fig_09_trend_type_breakdown.png")

    # Binary breakdown of would-route-to-edge-case (use_gpt flag in TrendDetector)
    edge_counts = tr_df["use_gpt_flag"].value_counts()
    plt.figure(figsize=(6, 4.5))
    sns.barplot(x=["ML route", "Edge-case route"], y=[edge_counts.get(False, 0), edge_counts.get(True, 0)], palette=["#2ca02c", "#ff7f0e"])
    plt.ylabel("Companies (last-4Q regime)")
    plt.title("Hybrid Routing: ML vs Edge-case (proxy)")
    _savefig("fig_10_hybrid_routing_breakdown.png")


def main() -> None:
    sns.set_theme(style="whitegrid")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    df = load_data()
    model_meta = load_model_metadata()

    make_missingness_top30(df)
    make_basic_distributions(df)
    make_geography_and_sector(df)
    make_model_performance(model_meta)
    make_feature_importance(model_meta)
    make_trend_routing_breakdown(df)

    # Optionally copy legacy plots if they exist (archive/plots)
    legacy_dir = ROOT / "archive" / "plots"
    if legacy_dir.exists():
        for name in [
            "model_performance.png",
            "feature_importance.png",
            "training_data_growth_analysis.png",
            "qoq_development_analysis.png",
            "bvp_cloud_index_performance.png",
        ]:
            src = legacy_dir / name
            dst = FIG_DIR / f"legacy_{name}"
            if src.exists() and not dst.exists():
                try:
                    dst.write_bytes(src.read_bytes())
                except Exception:
                    pass

    print(f"✅ Figures generated in: {FIG_DIR}")


if __name__ == "__main__":
    main()

