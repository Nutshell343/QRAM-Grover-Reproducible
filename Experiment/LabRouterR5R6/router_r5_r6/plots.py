"""Compact diagnostic plots; plotting errors never discard raw experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure).print_png(str(path))
    return path


def readout_plot(report: dict[str, Any], path: Path) -> Path:
    values = np.asarray(report["local_A"])
    figure = Figure(figsize=(11, 3), dpi=150, layout="constrained")
    for wire in range(4):
        axis = figure.add_subplot(1, 4, wire + 1)
        image = axis.imshow(values[wire], vmin=0, vmax=1, cmap="Blues")
        axis.set_title(f"Q{wire + 1}  cond={report['local_condition'][wire]:.1f}")
        axis.set_xticks(range(3), ["0", "1", "2"])
        axis.set_yticks(range(3), ["I", "X01", "XA"])
        axis.set_xlabel("Prep")
    figure.colorbar(image, ax=figure.axes, fraction=0.025, label="P(0)")
    return save(figure, path)


def paired_plot(report: dict[str, Any], path: Path) -> Path:
    figure = Figure(figsize=(10, 3.4), dpi=150, layout="constrained")
    axis = figure.add_subplot(1, 3, 1)
    axis.bar(("X", "Y", "Z"), report["conditional_bloch_linear"], color="tab:blue")
    axis.set_ylim(-1.1, 1.1)
    axis.set_ylabel("Conditional Bloch")
    axis.set_title("Output for |0>")
    labels = ("0", "1", "+", "+i")
    survival = [report["input_states"][label]["conditional_survival_linear"] for label in labels]
    residual = [report["unitary_residual"][label]["bloch_residual_norm"] for label in labels]
    axis = figure.add_subplot(1, 3, 2)
    axis.bar(labels, survival, color="tab:green")
    axis.set_ylim(0, 1.05)
    axis.set_title("Aux. survival")
    axis = figure.add_subplot(1, 3, 3)
    axis.bar(labels, residual, color="tab:red")
    axis.set_title("Unitary residual")
    figure.suptitle(f"R5 {report['direction']} paired block", fontsize=12)
    return save(figure, path)


def truth_plot(stage: str, report: dict[str, Any], path: Path) -> Path:
    words = list(report["inputs"])
    rows = [report["inputs"][word] for word in words]
    values = np.array([[item["correct_probability_linear"],
                        item["target_survival_linear"] - item["correct_probability_linear"],
                        item["computational_survival_linear"] - item["target_survival_linear"],
                        1 - item["computational_survival_linear"]] for item in rows])
    figure = Figure(figsize=(8, 3.7), dpi=150, layout="constrained")
    axis = figure.add_subplot(111)
    left = np.zeros(len(rows))
    for column, (name, color) in enumerate(zip(
        ("Correct", "Other target", "Other comp.", "Level 2"),
        ("tab:blue", "tab:orange", "tab:purple", "tab:red"))):
        axis.barh(range(len(rows)), values[:, column], left=left, label=name, color=color)
        left += values[:, column]
    axis.set_yticks(range(len(rows)), [f"{word}→{rows[i]['target']}" for i, word in enumerate(words)])
    axis.invert_yaxis()
    axis.set_xlabel("Probability (linear inversion)")
    axis.set_title(f"{stage} | FZ={report['F_Z']:.3f}, L={report['L']:.3f}")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=8)
    return save(figure, path)


def ramsey_plot(stage: str, report: dict[str, Any], path: Path) -> Path:
    figure = Figure(figsize=(10, 3.1), dpi=150, layout="constrained")
    for index, fit in enumerate(report["pairs"]):
        axis = figure.add_subplot(1, 3, index + 1)
        x = np.asarray(fit["scan_phase_rad"])
        axis.errorbar(x, fit["scan_contrast"], yerr=fit["scan_standard_error"],
                      fmt="o", ms=3, label="Data")
        axis.plot(x, fit["fit_contrast"], color="tab:red", label="Fit")
        axis.set_title(f"{('c', 'l', 'r')[index]}: {fit['phase_rad']:+.2f} rad")
        axis.set_xlabel("Scan phase (rad)")
        if index == 0:
            axis.set_ylabel("Binary contrast")
        axis.grid(alpha=0.2)
    return save(figure, path)


def fidelity_plot(stage: str, report: dict[str, Any], path: Path) -> Path:
    figure = Figure(figsize=(6, 3.4), dpi=150, layout="constrained")
    axis = figure.add_subplot(111)
    if stage.endswith("bound"):
        labels = ("FZ", "FX", "Fe low", "Fe high", "Leak")
        values = (report["F_Z"], report["F_X"], report["F_e_lower_linear"],
                  report["F_e_upper_linear"], report["L"])
    else:
        labels = ("FZ", "Fe", "Favg", "Leak")
        values = (report["F_Z"], report["F_e_linear"], report["F_avg_linear"], report["L"])
    axis.bar(labels, values, color=("tab:blue", "tab:green", "tab:cyan", "tab:red", "tab:purple")[:len(labels)])
    axis.set_ylim(0, max(1.05, max(values) + 0.05))
    axis.set_ylabel("Unconditional estimate")
    axis.set_title(stage)
    return save(figure, path)


def query_plot(report: dict[str, Any], path: Path) -> Path:
    values = np.array([[report["inputs"][f"{table}:{source}"]["correct_probability_linear"]
                        for source in ("0000", "0100", "1000", "1100")]
                       for table in ("00", "01", "10", "11")])
    figure = Figure(figsize=(5.5, 3.5), dpi=150, layout="constrained")
    axis = figure.add_subplot(111)
    image = axis.imshow(values, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(4), ("0000", "0100", "1000", "1100"))
    axis.set_yticks(range(4), ("00", "01", "10", "11"))
    axis.set_xlabel("Input")
    axis.set_ylabel("Memory")
    axis.set_title("Query target probability")
    figure.colorbar(image, ax=axis)
    return save(figure, path)


def render_stage(stage: str, report: dict[str, Any], output: Path) -> list[Path]:
    path = output / f"{stage}.png"
    if stage == "readout_local":
        return [readout_plot(report, path)]
    if stage == "paired":
        return [paired_plot(report, path)]
    if stage in ("truth_pilot", "truth"):
        return [truth_plot(stage, report, path)]
    if stage in ("ramsey_raw", "ramsey_dither", "ramsey_corrected"):
        return [ramsey_plot(stage, report, path)]
    if stage.endswith(("bound", "direct")):
        return [fidelity_plot(stage, report, path)]
    if stage == "query":
        return [query_plot(report, path)]
    return []


def suite_plot(rows: list[dict[str, Any]], path: Path) -> Path:
    figure = Figure(figsize=(9, 4), dpi=150, layout="constrained")
    labels = [f"{row['scheme'].upper()} {row['direction']}" for row in rows]
    axis = figure.add_subplot(1, 2, 1)
    axis.bar(labels, [row.get("truth", {}).get("F_Z", float("nan")) for row in rows],
             color="tab:blue")
    axis.set_ylim(0, 1.05)
    axis.set_title("Target population")
    axis.tick_params(axis="x", rotation=25)
    axis = figure.add_subplot(1, 2, 2)
    axis.bar(labels, [row.get("truth", {}).get("L", float("nan")) for row in rows],
             color="tab:red")
    axis.set_title("Level-2 leakage")
    axis.tick_params(axis="x", rotation=25)
    return save(figure, path)
