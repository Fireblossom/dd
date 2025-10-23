#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


CURVES_DIR = os.path.dirname(os.path.abspath(__file__))


def load_annotated_curves(csv_paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV: {path}: {exc}") from exc

        required_cols = {
            "dataset",
            "labels_micro_f1",
            "openness_f1",
            "model",
            "step",
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns {sorted(missing)} in {path}")

        # Normalize types
        df = df.copy()
        df["model"] = df["model"].astype(str)
        df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["step"]).astype({"step": int})

        # Map dataset to a simpler type: label vs phrase
        def map_dataset_type(value: str) -> str:
            if isinstance(value, str) and "withphrase" in value:
                return "phrase"
            return "label"

        df["dataset_type"] = df["dataset"].map(map_dataset_type)

        frames.append(df[[
            "model",
            "dataset_type",
            "step",
            "labels_micro_f1",
            "openness_f1",
        ]])

    all_df = pd.concat(frames, ignore_index=True)

    # Sort by step within each series for clean lines
    all_df = all_df.sort_values(["model", "dataset_type", "step"]).reset_index(drop=True)
    return all_df


def plot_metric(
    data: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: str,
    model_colors: Dict[str, str],
    line_styles: Dict[str, str],
) -> None:
    plt.figure(figsize=(8.5, 5.0))

    # Unique series present in data
    for model in sorted(data["model"].unique()):
        for dataset_type in ["label", "phrase"]:
            series = data[(data["model"] == model) & (data["dataset_type"] == dataset_type)]
            if series.empty:
                continue

            x = series["step"].values
            y = series[metric_col].values

            color = model_colors.get(model, "#555555")
            linestyle = line_styles.get(dataset_type, "-")

            label = f"{model} {dataset_type}"
            plt.plot(
                x,
                y,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker=None,
            )

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(metric_col)
    plt.grid(True, alpha=0.25, linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_grid(
    data: pd.DataFrame,
    output_path: str,
    model_colors: Dict[str, str],
    line_styles: Dict[str, str],
):
    # Models row order and pretty names to match paper wording
    model_order = ["gemma", "yi", "qwen"]
    pretty_names = {"gemma": "Gemma-3", "yi": "Yi-1.5", "qwen": "Qwen-3"}
    metrics = [
        ("labels_micro_f1", "Localness Criteria — Micro F1"),
        ("openness_f1", "Openness Tendency — F1"),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9), sharex=True, sharey=False)

    def smooth_series(values, window: int = 3):
        if values is None:
            return None
        import numpy as np
        arr = np.asarray(values, dtype=float)
        if arr.size < 2 or window <= 1:
            return arr
        if window > arr.size:
            window = arr.size
        kernel = np.ones(window, dtype=float) / float(window)
        pad = window // 2  # works best with odd window
        # Edge padding avoids zero-padding artifacts at the ends
        padded = np.pad(arr, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed

    for r, model in enumerate(model_order):
        model_df = data[data["model"] == model]
        for c, (metric_col, col_title) in enumerate(metrics):
            ax = axes[r, c]
            # Align two series by step and plot smoothed lines with shaded difference
            df_label = (
                model_df[model_df["dataset_type"] == "label"][["step", metric_col]].rename(columns={metric_col: f"{metric_col}_label"})
            )
            df_phrase = (
                model_df[model_df["dataset_type"] == "phrase"][["step", metric_col]].rename(columns={metric_col: f"{metric_col}_phrase"})
            )
            merged = pd.merge(df_label, df_phrase, on="step", how="inner").sort_values("step")
            if not merged.empty:
                x = merged["step"].values
                y_label = smooth_series(merged[f"{metric_col}_label"].values, window=3)
                y_phrase = smooth_series(merged[f"{metric_col}_phrase"].values, window=3)

                color = model_colors.get(model, "#555555")

                ax.plot(
                    x,
                    y_label,
                    label="Labels-only",
                    color=color,
                    linestyle=line_styles.get("label", "--"),
                    linewidth=2.0,
                )
                ax.plot(
                    x,
                    y_phrase,
                    label="Labels-with-phrase",
                    color=color,
                    linestyle=line_styles.get("phrase", "-"),
                    linewidth=2.0,
                )
                # Shaded area between the two curves
                import numpy as np
                lower = np.minimum(y_label, y_phrase)
                upper = np.maximum(y_label, y_phrase)
                ax.fill_between(x, lower, upper, color=color, alpha=0.15, linewidth=0)

            # Row titles (model names)
            ax.set_ylabel("Score" if c == 0 else "", fontsize=10)
            if c == 0:
                ax.set_title(col_title if r == 0 else "")
            else:
                ax.set_title(col_title if r == 0 else "")

            # Left gutter can show model pretty name as text annotation
            ax.text(0.01, 0.95, pretty_names.get(model, model), transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
            ax.grid(True, alpha=0.25, linewidth=0.8)

    # X labels only on bottom row
    for c in range(2):
        axes[-1, c].set_xlabel("Checkpoint Step")

    # Column titles on top row already set; create a single shared legend with BLACK line samples
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label="Labels-only"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="Labels-with-phrase"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=2, frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    csv_paths = sorted(glob.glob(os.path.join(CURVES_DIR, "annotated - *.csv")))
    if not csv_paths:
        raise FileNotFoundError("No CSVs found matching 'annotated - *.csv' in curves directory")

    df = load_annotated_curves(csv_paths)

    # Define styling: colors per model, linestyles per dataset_type
    model_colors = {
        "gemma": "#6f42c1",  # purple
        "yi": "#2ca02c",     # green
        "qwen": "#ff7f0e",   # orange
    }
    line_styles = {
        "label": "--",   # dashed
        "phrase": "-",   # solid
    }

    # Labels plot: labels_micro_f1
    labels_out = os.path.join(CURVES_DIR, "annotated_curves_labels.png")
    plot_metric(
        data=df,
        metric_col="labels_micro_f1",
        title="Labels (micro F1)",
        output_path=labels_out,
        model_colors=model_colors,
        line_styles=line_styles,
    )

    # Openness plot: openness_f1
    openness_out = os.path.join(CURVES_DIR, "annotated_curves_openness.png")
    plot_metric(
        data=df,
        metric_col="openness_f1",
        title="Openness (F1)",
        output_path=openness_out,
        model_colors=model_colors,
        line_styles=line_styles,
    )

    # 3x2 Grid plot
    grid_out = os.path.join(CURVES_DIR, "annotated_curves_grid.png")
    plot_grid(
        data=df,
        output_path=grid_out,
        model_colors=model_colors,
        line_styles=line_styles,
    )

    print(f"[OK] Saved: {labels_out}")
    print(f"[OK] Saved: {openness_out}")
    print(f"[OK] Saved: {grid_out}")


if __name__ == "__main__":
    main()


