"""Run the Adam vs AdaMax experiment grid on MNIST.

    python -m adam_study.experiments --seeds 5 --epochs 20 --out results

The default grid has ten configurations: logistic regression, and the MLP and CNN with
and without dropout, each trained with Adam and with AdaMax. Under ``--out`` it writes:

* ``results.json``: per-epoch metrics of every run, rewritten after each run. Runs that
  are already in the file are skipped, so an interrupted invocation resumes where it
  stopped.
* ``summary.csv`` and ``summary.md``: final-epoch metrics per configuration, as mean
  and sample standard deviation over seeds.
* ``figures/*.png``: the figures from :mod:`adam_study.plots`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from adam_study.data import load_mnist, to_device
from adam_study.models import MODEL_LABELS, MODEL_NAMES
from adam_study.plots import save_figures
from adam_study.train import (
    OPTIMIZER_LABELS,
    OPTIMIZERS,
    SCHEDULES,
    RunConfig,
    fit,
    resolve_device,
)

log = logging.getLogger("adam_study")

Run = dict[str, Any]
METRICS = ("test_acc", "test_loss", "train_loss")


def grid(
    models: Sequence[str], optimizers: Sequence[str], dropouts: Sequence[float]
) -> list[tuple[str, str, float]]:
    """(model, optimizer, dropout) for every configuration, in table order, without repeats.

    Logistic regression has no hidden layer to apply dropout to, so it only runs with 0.
    """
    configs = (
        (model, optimizer, dropout)
        for model in models
        for dropout in dropouts
        if model != "logreg" or dropout == 0
        for optimizer in optimizers
    )
    return list(dict.fromkeys(configs))


def load_runs(path: Path, settings: dict[str, Any] | None = None) -> list[Run]:
    """Read the runs saved in ``path``; with ``settings``, refuse a file from other settings."""
    if not path.exists():
        return []
    saved = json.loads(path.read_text())
    if settings is not None and saved["settings"] != settings:
        raise SystemExit(
            f"{path} holds runs made with {saved['settings']}, not {settings}. "
            "Use another --out directory for different settings."
        )
    return saved["runs"]


def save_runs(path: Path, settings: dict[str, Any], runs: list[Run]) -> None:
    """Write the runs atomically, so an interruption never leaves a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"settings": settings, "runs": runs}, indent=1))
    os.replace(tmp, path)


def table_order(model: str, dropout: float, optimizer: str = "adam") -> tuple[int, float, int]:
    """Sort key that lists configurations as in the README: by model, dropout, optimizer."""
    return MODEL_NAMES.index(model), dropout, list(OPTIMIZERS).index(optimizer)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    return mean, statistics.stdev(values) if len(values) > 1 else 0.0


def format_mean_std(values: Sequence[float], signed: bool = False) -> str:
    """Format as ``mean ± sample std`` with two decimals, or the value alone if there is one.

    With ``signed``, a mean that does not round to zero gets an explicit sign.
    """
    mean, std = _mean_std(values)
    if round(mean, 2) == 0:
        mean, signed = 0.0, False  # "0.00" rather than "-0.00" or "+0.00"
    text = f"{mean:+.2f}" if signed else f"{mean:.2f}"
    return f"{text} ± {std:.2f}" if len(values) > 1 else text


def summarize(runs: Sequence[Run]) -> list[dict[str, Any]]:
    """Final-epoch metrics per configuration: mean and sample std over seeds."""
    groups: dict[tuple[str, float, str], list[Run]] = {}
    for run in runs:
        groups.setdefault((run["model"], run["dropout"], run["optimizer"]), []).append(run)
    rows = []
    for config in sorted(groups, key=lambda c: table_order(*c)):
        model, dropout, optimizer = config
        row: dict[str, Any] = {
            "model": model,
            "dropout": dropout,
            "optimizer": optimizer,
            "seeds": len(groups[config]),
        }
        for metric in METRICS:
            mean, std = _mean_std([run[metric][-1] for run in groups[config]])
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
        rows.append(row)
    return rows


def accuracy_table(runs: Sequence[Run]) -> str:
    """Markdown table of final test accuracy (%), one row per model and dropout setting.

    When both optimizers are present, the last column is the per-seed difference
    Adam - AdaMax. Runs with the same seed share initial weights and minibatch order, so
    the difference isolates the optimizer.
    """
    optimizers = [name for name in OPTIMIZERS if any(r["optimizer"] == name for r in runs)]
    paired = len(optimizers) == 2
    header = ["Model", "Dropout", *(OPTIMIZER_LABELS[name] for name in optimizers)]
    if paired:
        header.append("Adam - AdaMax")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    configs = sorted({(r["model"], r["dropout"]) for r in runs}, key=lambda c: table_order(*c))
    for model, dropout in configs:
        by_optimizer = {
            name: {
                r["seed"]: r["test_acc"][-1]
                for r in runs
                if (r["model"], r["dropout"], r["optimizer"]) == (model, dropout, name)
            }
            for name in optimizers
        }
        cells = [MODEL_LABELS[model], f"{dropout:g}" if dropout else "none"]
        for name in optimizers:
            accs = list(by_optimizer[name].values())
            cells.append(format_mean_std(accs) if accs else "")
        if paired:
            adam, adamax = by_optimizer["adam"], by_optimizer["adamax"]
            diffs = [adam[seed] - adamax[seed] for seed in sorted(adam.keys() & adamax.keys())]
            cells.append(format_mean_std(diffs, signed=True) if diffs else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summaries(out: Path, runs: Sequence[Run]) -> None:
    """Write ``summary.csv`` and ``summary.md`` for the runs so far."""
    rows = summarize(runs)
    if not rows:
        return
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})
    epochs = len(runs[0]["test_acc"])
    (out / "summary.md").write_text(
        f"Final test accuracy (%) after {epochs} epoch{'s' * (epochs != 1)}, "
        "mean ± std over seeds.\n\n" + accuracy_table(runs) + "\n"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m adam_study.experiments",
        description="Train every configuration of the Adam vs AdaMax study on MNIST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5, help="seeds 0..N-1 per configuration")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs per run")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST location")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="models to train",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=list(OPTIMIZERS),
        default=list(OPTIMIZERS),
        help="optimizers to compare",
    )
    parser.add_argument(
        "--dropout",
        nargs="+",
        type=float,
        default=[0.0, 0.2],
        help="dropout rates for the MLP and CNN (logistic regression runs only with 0)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="step size alpha")
    parser.add_argument("--batch-size", type=int, default=128, help="minibatch size")
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default="constant",
        help="'inv_sqrt' decays the step size as alpha / sqrt(t) over updates t; "
        "it needs a larger --lr than 'constant'",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="coupled L2 penalty coefficient"
    )
    parser.add_argument("--device", default="auto", help="'auto' (CUDA if available), 'cpu', ...")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="redraw summaries and figures from an existing results.json without training",
    )
    args = parser.parse_args(argv)
    # Checked before any training, so an invalid value never leaves partial results.
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not all(0.0 <= p < 1.0 for p in args.dropout):
        parser.error("--dropout rates must be in [0, 1)")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    settings = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "schedule": args.schedule,
        "weight_decay": args.weight_decay,
    }

    if args.plots_only:
        runs = load_runs(results_path)
        if not runs:
            raise SystemExit(f"no runs found in {results_path}")
    else:
        runs = load_runs(results_path, settings)
        done = {(r["model"], r["optimizer"], r["dropout"], r["seed"]) for r in runs}
        configs = grid(args.models, args.optimizers, args.dropout)
        # Seeds form the outer loop, so a partial run still covers every configuration.
        todo = [
            (model, optimizer, dropout, seed)
            for seed in range(args.seeds)
            for model, optimizer, dropout in configs
            if (model, optimizer, dropout, seed) not in done
        ]
        log.info(
            "%d runs requested, %d already in %s",
            len(configs) * args.seeds,
            len(configs) * args.seeds - len(todo),
            results_path,
        )
        if todo:
            device = resolve_device(args.device)
            log.info("training on %s with torch %s", device, torch.__version__)
            train_set, test_set = load_mnist(args.data_dir)
            if device.type != "cpu":
                # MNIST fits in device memory; keeping it there avoids a copy per batch.
                train_set, test_set = to_device(train_set, device), to_device(test_set, device)
            for i, (model, optimizer, dropout, seed) in enumerate(todo, start=1):
                config = RunConfig(
                    model=model, optimizer=optimizer, dropout=dropout, seed=seed, **settings
                )
                log.info("run %d/%d: %s", i, len(todo), config.label)
                start = time.perf_counter()
                history = fit(config, train_set, test_set, device)
                runs.append(
                    {
                        "model": model,
                        "optimizer": optimizer,
                        "dropout": dropout,
                        "seed": seed,
                        **asdict(history),
                        "seconds": round(time.perf_counter() - start, 1),
                        "device": device.type,
                        "torch": torch.__version__,
                    }
                )
                save_runs(results_path, settings, runs)
                write_summaries(args.out, runs)

    if not runs:
        log.warning("no runs to summarize")
        return
    write_summaries(args.out, runs)
    log.info("final test accuracy (%%):\n%s", accuracy_table(runs))
    if not args.no_plots:
        for path in save_figures(runs, args.out / "figures"):
            log.info("wrote %s", path)


if __name__ == "__main__":
    main()
