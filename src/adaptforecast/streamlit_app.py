"""Local interactive UI with a safe precomputed-results fallback."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from adaptforecast.config import load_config
from adaptforecast.data import load_canonical
from adaptforecast.schema import DataValidationError, validate_dataframe

ROOT = Path(__file__).resolve().parents[2]


def available_runs() -> list[Path]:
    manifests = list((ROOT / "artifacts").glob("*/manifest.json"))
    manifests.extend((ROOT / "data" / "precomputed").glob("**/manifest.json"))
    return sorted(manifests, key=lambda path: path.stat().st_mtime, reverse=True)


def show_run(manifest_path: Path) -> None:
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    st.subheader(f"Experiment: {run_dir.name}")
    st.json(manifest)
    metrics_path = run_dir / "metrics.csv"
    predictions_path = run_dir / "predictions.csv"
    rules = sorted(run_dir.glob("efs/**/rules.csv"))
    if metrics_path.exists() and metrics_path.stat().st_size:
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)
    if predictions_path.exists() and predictions_path.stat().st_size:
        predictions = pd.read_csv(predictions_path, parse_dates=["date"])
        if not predictions.empty:
            category = st.selectbox("Category", sorted(predictions["category"].unique()))
            selected = predictions.loc[predictions["category"] == category].copy()
            selected["series"] = selected["model"] + ":" + selected["variant"]
            chart = selected.pivot_table(index="date", columns="series", values="predicted")
            actual = selected.groupby("date")["actual"].first().rename("actual")
            st.line_chart(pd.concat([actual, chart], axis=1))
    if rules:
        st.subheader("Fuzzy-rule audit")
        st.caption("Rules express learned associations, not causal claims.")
        selected_rules = st.selectbox(
            "Rule artifact", rules, format_func=lambda path: str(path.relative_to(run_dir))
        )
        st.dataframe(pd.read_csv(selected_rules), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AdaptForecast", layout="wide")
    st.title("AdaptForecast / EFS-forecasting")
    st.write(
        "Auditable next-day category demand forecasting for small datasets using "
        "interval type-2 fuzzy systems."
    )
    matlab_ready = shutil.which("matlab") is not None
    if matlab_ready:
        st.success("MATLAB detected: EFS training can be launched locally.")
    else:
        st.info("MATLAB was not detected. The app is in precomputed-results browsing mode.")

    with st.sidebar:
        st.header("Data validation")
        upload = st.file_uploader("Canonical CSV", type=["csv"])
        if upload is not None:
            try:
                frame = pd.read_csv(upload)
                report = validate_dataframe(frame)
                st.success(f"Valid: {report.rows} rows")
                st.json(report.to_dict())
            except (DataValidationError, ValueError) as exc:
                st.error(str(exc))

    tabs = st.tabs(["Audited runs", "Sample data", "Run locally"])
    with tabs[0]:
        runs = available_runs()
        if not runs:
            st.warning("No audited artifact runs are available yet.")
        else:
            chosen = st.selectbox("Run", runs, format_func=lambda path: path.parent.name)
            show_run(chosen)
    with tabs[1]:
        sample = ROOT / "data" / "sample" / "synthetic_demand.csv"
        data = load_canonical(sample)
        st.dataframe(data.head(100), use_container_width=True)
        st.caption("This dataset is synthetic and is not derived from the private Walmart data.")
    with tabs[2]:
        profile = st.radio("Profile", ["smoke", "full"], horizontal=True)
        sample_categories = sorted(
            load_canonical(ROOT / "data" / "sample" / "synthetic_demand.csv")["category"].unique()
        )
        run_category = st.selectbox("Training category", sample_categories)
        st.write(
            "Training is intentionally explicit and writes a complete artifact directory. "
            "Full EFS training requires MATLAB R2024b+, Fuzzy Logic Toolbox, Global "
            "Optimization Toolbox, and optionally Parallel Computing Toolbox."
        )
        if st.button("Run benchmark", disabled=not matlab_ready):
            from adaptforecast.benchmark import run_benchmark

            config_name = "benchmark.smoke.yaml" if profile == "smoke" else "benchmark.yaml"
            config = load_config(ROOT / "configs" / config_name)
            config.categories = [run_category]
            with st.status("Running audited benchmark...", expanded=True):
                run_dir = run_benchmark(config, ROOT)
            st.success(f"Completed: {run_dir}")


if __name__ == "__main__":
    main()
