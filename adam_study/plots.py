"""Figures for the study.

Every curve is the per-epoch mean over seeds, with a shaded band of plus or minus one
sample standard deviation. Color identifies the optimizer (Adam blue, AdaMax orange) in
every figure. In the summary and dropout figures, line style marks the dropout rate (solid
without dropout, dashed with p = 0.2); the per-model figures put each dropout setting in
its own row. Loss axes use a log scale when the curves on them span more than a factor of
ten.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

from adam_study.models import MODEL_LABELS, MODEL_NAMES
from adam_study.train import OPTIMIZER_LABELS

Run = dict[str, Any]

COLORS = {"adam": "#2a78d6", "adamax": "#eb6834"}
METRIC_LABELS = {
    "train_loss": "Training loss",
    "test_loss": "Test loss",
    "test_acc": "Test accuracy (%)",
}
LOG_SCALE_METRICS = {"train_loss", "test_loss"}

_INK, _SECONDARY_INK, _AXIS, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9", "#fcfcfb"
STYLE = {
    "figure.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "axes.edgecolor": _AXIS,
    "axes.labelcolor": _SECONDARY_INK,
    "axes.titlecolor": _INK,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.6,
    "xtick.color": _AXIS,
    "ytick.color": _AXIS,
    "xtick.labelcolor": _SECONDARY_INK,
    "ytick.labelcolor": _SECONDARY_INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.size": 9,
    "figure.titlesize": 11,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
}


def _present(runs: Iterable[Run], key: str, order: Iterable[str]) -> list[str]:
    """Values of ``key`` that occur in ``runs``, in the canonical ``order``."""
    found = {run[key] for run in runs}
    return [value for value in order if value in found]


def _select(runs: Iterable[Run], model: str, optimizer: str, dropout: float) -> list[Run]:
    return [
        run
        for run in runs
        if run["model"] == model and run["optimizer"] == optimizer and run["dropout"] == dropout
    ]


def _linestyle(dropout: float, dropouts: Sequence[float]) -> str:
    """Solid without dropout; dashed, dotted, ... for the dropout rates in increasing order."""
    rates = [p for p in sorted(dropouts) if p > 0]
    return "-" if dropout == 0 else ("--", ":", "-.")[rates.index(dropout) % 3]


def _dropout_label(dropout: float) -> str:
    return f"dropout {dropout:g}" if dropout else "no dropout"


def _seed_note(runs: Sequence[Run]) -> str:
    counts = set(Counter((r["model"], r["optimizer"], r["dropout"]) for r in runs).values())
    if len(counts) != 1:
        return "mean ± std over seeds"
    n = counts.pop()
    return "1 seed" if n == 1 else f"mean ± std over {n} seeds"


def _plot_mean_std(ax: Axes, runs: Sequence[Run], metric: str, color: str, linestyle: str) -> None:
    values = np.array([run[metric] for run in runs], dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    epochs = np.arange(1, mean.size + 1)
    ax.plot(epochs, mean, color=color, linestyle=linestyle, marker="o" if mean.size == 1 else None)
    # On a log axis a band edge at or below zero is clipped to the bottom of the plot.
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)


def _spans_a_decade(ax: Axes) -> bool:
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()])
    values = values[values > 0]
    return values.size > 0 and values.max() > 10 * values.min()


def _format_axes(ax: Axes, metric: str, xlabel: bool = True, ylabel: bool = True) -> None:
    """Label the axes; call after plotting, since the y scale depends on the data."""
    if metric in LOG_SCALE_METRICS and _spans_a_decade(ax):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10], min_n_ticks=1))
    if xlabel:
        ax.set_xlabel("Epoch")
    if ylabel:
        ax.set_ylabel(METRIC_LABELS[metric])


def _legend_handles(optimizers: Iterable[str], dropouts: Sequence[float]) -> list[Line2D]:
    handles = [Line2D([], [], color=COLORS[o], label=OPTIMIZER_LABELS[o]) for o in optimizers]
    if len(dropouts) > 1:
        handles += [
            Line2D(
                [],
                [],
                color=_SECONDARY_INK,
                linestyle=_linestyle(p, dropouts),
                label=_dropout_label(p),
            )
            for p in dropouts
        ]
    return handles


@rc_context(STYLE)
def summary_figure(runs: Sequence[Run]) -> Figure | None:
    """Training loss and test accuracy for every model and configuration in one figure."""
    models = _present(runs, "model", MODEL_NAMES)
    if not models:
        return None
    optimizers = _present(runs, "optimizer", OPTIMIZER_LABELS)
    dropouts = sorted({run["dropout"] for run in runs})

    fig = Figure(figsize=(3.5 * len(models) + 0.6, 5.6), layout="constrained")
    axes = fig.subplots(2, len(models), squeeze=False)
    for col, model in enumerate(models):
        for row, metric in enumerate(("train_loss", "test_acc")):
            ax = axes[row, col]
            for dropout in dropouts:
                for optimizer in optimizers:
                    selected = _select(runs, model, optimizer, dropout)
                    if selected:
                        style = _linestyle(dropout, dropouts)
                        _plot_mean_std(ax, selected, metric, COLORS[optimizer], style)
            _format_axes(ax, metric, xlabel=row == 1, ylabel=col == 0)
        axes[0, col].set_title(MODEL_LABELS[model])

    fig.suptitle(f"Adam vs AdaMax on MNIST ({_seed_note(runs)})")
    fig.legend(handles=_legend_handles(optimizers, dropouts), loc="outside lower center", ncols=4)
    return fig


@rc_context(STYLE)
def optimizer_figure(runs: Sequence[Run], model: str) -> Figure | None:
    """Adam vs AdaMax for one model: training loss and test accuracy per dropout setting."""
    model_runs = [run for run in runs if run["model"] == model]
    if not model_runs:
        return None
    optimizers = _present(model_runs, "optimizer", OPTIMIZER_LABELS)
    dropouts = sorted({run["dropout"] for run in model_runs})

    fig = Figure(figsize=(8, 0.6 + 3.0 * len(dropouts)), layout="constrained")
    axes = fig.subplots(len(dropouts), 2, squeeze=False)
    for row, dropout in enumerate(dropouts):
        for col, metric in enumerate(("train_loss", "test_acc")):
            ax = axes[row, col]
            for optimizer in optimizers:
                selected = _select(model_runs, model, optimizer, dropout)
                if selected:
                    _plot_mean_std(ax, selected, metric, COLORS[optimizer], "-")
            _format_axes(ax, metric, xlabel=row == len(dropouts) - 1)
            if len(dropouts) > 1:
                ax.set_title(_dropout_label(dropout).capitalize())

    fig.suptitle(f"{MODEL_LABELS[model]}: Adam vs AdaMax ({_seed_note(model_runs)})")
    fig.legend(handles=_legend_handles(optimizers, [0.0]), loc="outside lower center", ncols=2)
    return fig


@rc_context(STYLE)
def dropout_figure(runs: Sequence[Run]) -> Figure | None:
    """With vs without dropout: training loss, test loss and test accuracy per model."""
    models = [m for m in MODEL_NAMES if len({r["dropout"] for r in runs if r["model"] == m}) > 1]
    if not models:
        return None
    model_runs = [run for run in runs if run["model"] in models]
    optimizers = _present(model_runs, "optimizer", OPTIMIZER_LABELS)
    dropouts = sorted({run["dropout"] for run in model_runs})

    fig = Figure(figsize=(11, 0.6 + 3.0 * len(models)), layout="constrained")
    axes = fig.subplots(len(models), 3, squeeze=False)
    for row, model in enumerate(models):
        for col, metric in enumerate(("train_loss", "test_loss", "test_acc")):
            ax = axes[row, col]
            for optimizer in optimizers:
                for dropout in dropouts:
                    selected = _select(model_runs, model, optimizer, dropout)
                    if selected:
                        style = _linestyle(dropout, dropouts)
                        _plot_mean_std(ax, selected, metric, COLORS[optimizer], style)
            _format_axes(ax, metric, xlabel=row == len(models) - 1)
            ax.set_title(MODEL_LABELS[model])

    fig.suptitle(f"Dropout vs no dropout ({_seed_note(model_runs)})")
    fig.legend(handles=_legend_handles(optimizers, dropouts), loc="outside lower center", ncols=4)
    return fig


def make_figures(runs: Sequence[Run]) -> dict[str, Figure]:
    """Build every figure the runs allow, keyed by file stem."""
    figures = {"summary": summary_figure(runs), "dropout": dropout_figure(runs)}
    for model in MODEL_NAMES:
        figures[f"adam_vs_adamax_{model}"] = optimizer_figure(runs, model)
    return {name: fig for name, fig in figures.items() if fig is not None}


def save_figures(runs: Sequence[Run], out_dir: str | Path, dpi: int = 150) -> list[Path]:
    """Write every figure as a PNG to ``out_dir`` and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in make_figures(runs).items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    return paths
