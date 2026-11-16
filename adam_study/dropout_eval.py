"""Measure how much testing with dropout active changes the accuracy of the dropout runs.

    python -m adam_study.dropout_eval --out results

The 2022 version of this project never switched its models to evaluation mode, so it
tested the dropout models with dropout still active. This command retrains every dropout
run in ``results.json`` with the same settings and seed, and tests each final model in
two ways:

* with dropout off (evaluation mode), as :mod:`adam_study.experiments` does. With the
  same PyTorch build, hardware and number of threads, retraining gives the same model,
  so this accuracy matches ``results.json``;
* with dropout on (training mode), as the 2022 code did, once per pass over the test
  set. Each pass draws new dropout masks.

Under ``--out`` it writes ``dropout_eval.json``, rewritten after each run (runs that are
already in it are skipped), and ``dropout_eval.md``, a table of the final accuracies.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adam_study.data import load_mnist, make_loader, to_device
from adam_study.experiments import Run, format_mean_std, load_runs, save_runs, table_order
from adam_study.models import MODEL_LABELS
from adam_study.train import OPTIMIZER_LABELS, RunConfig, resolve_device, train_model

log = logging.getLogger("adam_study")


@torch.no_grad()
def accuracy_with_dropout(
    model: nn.Module, loader: DataLoader, device: torch.device, passes: int
) -> list[float]:
    """Return the accuracy in percent of each of ``passes`` passes, with dropout active.

    The masks come from the global random number generator, so seed it first for
    reproducible results.
    """
    model.train()
    accuracies = []
    for _ in range(passes):
        correct, count = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            count += len(y)
        accuracies.append(100.0 * correct / count)
    return accuracies


def evaluate_run(
    config: RunConfig,
    train_set: TensorDataset,
    test_set: TensorDataset,
    device: torch.device | str = "cpu",
    passes: int = 20,
    test_batch_size: int = 1000,
) -> Run:
    """Train the run described by ``config``, then test the final model with dropout off and on."""
    device = torch.device(device)
    start = time.perf_counter()
    model, history = train_model(config, train_set, test_set, device, test_batch_size)
    torch.manual_seed(config.seed)
    dropout_on = accuracy_with_dropout(
        model, make_loader(test_set, test_batch_size), device, passes
    )
    return {
        "model": config.model,
        "optimizer": config.optimizer,
        "dropout": config.dropout,
        "seed": config.seed,
        "test_acc_dropout_off": history.test_acc[-1],
        "test_acc_dropout_on": dropout_on,
        "seconds": round(time.perf_counter() - start, 1),
        "device": device.type,
        "torch": torch.__version__,
    }


def dropout_table(runs: Sequence[Run]) -> str:
    """Markdown table of final test accuracy (%) with dropout off and on, over seeds.

    Each run's accuracy with dropout on is first averaged over its passes. The last
    column is the difference off - on, computed run by run.
    """
    lines = [
        "| Model | Dropout | Optimizer | Dropout off | Dropout on | Off - on |",
        "|---|---|---|---|---|---|",
    ]
    configs = {(r["model"], r["dropout"], r["optimizer"]) for r in runs}
    for config in sorted(configs, key=lambda c: table_order(*c)):
        model, dropout, optimizer = config
        selected = [r for r in runs if (r["model"], r["dropout"], r["optimizer"]) == config]
        off = [r["test_acc_dropout_off"] for r in selected]
        on = [statistics.fmean(r["test_acc_dropout_on"]) for r in selected]
        diffs = [a - b for a, b in zip(off, on, strict=True)]
        lines.append(
            f"| {MODEL_LABELS[model]} | {dropout:g} | {OPTIMIZER_LABELS[optimizer]} "
            f"| {format_mean_std(off)} | {format_mean_std(on)} "
            f"| {format_mean_std(diffs, signed=True)} |"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m adam_study.dropout_eval",
        description="Retrain the dropout runs in results.json and test them with dropout "
        "off and on.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results"), help="directory that holds results.json"
    )
    parser.add_argument(
        "--passes", type=int, default=20, help="passes over the test set with dropout on"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST location")
    parser.add_argument("--device", default="auto", help="'auto' (CUDA if available), 'cpu', ...")
    args = parser.parse_args(argv)
    if args.passes < 1:
        parser.error("--passes must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    grid_path, out_path = args.out / "results.json", args.out / "dropout_eval.json"
    if not grid_path.exists():
        raise SystemExit(f"{grid_path} not found; run python -m adam_study.experiments first")
    grid = json.loads(grid_path.read_text())
    settings = {**grid["settings"], "passes": args.passes}
    if out_path.exists() and json.loads(out_path.read_text())["settings"] != settings:
        raise SystemExit(f"{out_path} was made with other settings; remove it to start again")
    runs = load_runs(out_path)

    done = {(r["model"], r["optimizer"], r["dropout"], r["seed"]) for r in runs}
    todo = [
        r
        for r in grid["runs"]
        if r["dropout"] > 0 and (r["model"], r["optimizer"], r["dropout"], r["seed"]) not in done
    ]
    if todo:
        device = resolve_device(args.device)
        log.info("training on %s with torch %s", device, torch.__version__)
        train_set, test_set = load_mnist(args.data_dir)
        if device.type != "cpu":
            train_set, test_set = to_device(train_set, device), to_device(test_set, device)
        for i, grid_run in enumerate(todo, start=1):
            config = RunConfig(
                model=grid_run["model"],
                optimizer=grid_run["optimizer"],
                dropout=grid_run["dropout"],
                seed=grid_run["seed"],
                **grid["settings"],
            )
            log.info("run %d/%d: %s", i, len(todo), config.label)
            run = evaluate_run(config, train_set, test_set, device, args.passes)
            log.info(
                "%s: dropout off %.2f%% (results.json: %.2f%%), dropout on %.2f%% "
                "(mean of %d passes)",
                config.label,
                run["test_acc_dropout_off"],
                grid_run["test_acc"][-1],
                statistics.fmean(run["test_acc_dropout_on"]),
                args.passes,
            )
            runs.append(run)
            save_runs(out_path, settings, runs)

    if not runs:
        log.warning("%s has no runs with dropout", grid_path)
        return
    table = dropout_table(runs)
    (args.out / "dropout_eval.md").write_text(
        "Final test accuracy (%) of the dropout runs, tested with dropout off and with "
        f"dropout on (mean of {args.passes} passes over the test set), mean ± std over "
        "seeds.\n\n" + table + "\n"
    )
    log.info("final test accuracy (%%):\n%s", table)


if __name__ == "__main__":
    main()
