"""Testing the dropout runs with dropout off and on, on synthetic data."""

from __future__ import annotations

import json

import pytest
import torch

from adam_study import dropout_eval, experiments
from adam_study.data import make_loader
from adam_study.models import build_model
from adam_study.train import evaluate

CPU = torch.device("cpu")


@pytest.fixture
def grid_dir(tmp_path, monkeypatch, digits):
    """An output directory holding a small MLP grid, with and without dropout."""
    for module in (experiments, dropout_eval):
        monkeypatch.setattr(module, "load_mnist", lambda *args, **kwargs: digits)
    args = ["--models", "mlp", "--seeds", "2", "--epochs", "2", "--batch-size", "64"]
    experiments.main([*args, "--out", str(tmp_path), "--device", "cpu", "--no-plots"])
    return tmp_path


def test_accuracy_changes_between_passes_only_with_dropout(digits):
    loader = make_loader(digits[1], batch_size=64)
    torch.manual_seed(0)
    model = build_model("mlp", dropout=0.5)
    accuracies = dropout_eval.accuracy_with_dropout(model, loader, CPU, passes=10)
    assert len(accuracies) == 10
    assert len(set(accuracies)) > 1

    model = build_model("mlp")
    accuracy = evaluate(model, loader, CPU)[1]
    assert dropout_eval.accuracy_with_dropout(model, loader, CPU, passes=3) == [accuracy] * 3


def test_retrains_each_dropout_run_and_tests_it_both_ways(grid_dir):
    dropout_eval.main(["--out", str(grid_dir), "--passes", "3", "--device", "cpu"])

    grid = json.loads((grid_dir / "results.json").read_text())
    saved = json.loads((grid_dir / "dropout_eval.json").read_text())
    assert saved["settings"] == {**grid["settings"], "passes": 3}
    final_acc = {
        (r["model"], r["optimizer"], r["dropout"], r["seed"]): r["test_acc"][-1]
        for r in grid["runs"]
    }
    keys = [(r["model"], r["optimizer"], r["dropout"], r["seed"]) for r in saved["runs"]]
    assert sorted(keys) == sorted(key for key in final_acc if key[2] > 0)
    for run, key in zip(saved["runs"], keys, strict=True):
        # Same settings, seed and machine, so the retrained model is the one in the grid.
        assert run["test_acc_dropout_off"] == final_acc[key]
        assert len(run["test_acc_dropout_on"]) == 3
    assert "| MLP | 0.2 | Adam |" in (grid_dir / "dropout_eval.md").read_text()


def test_skips_finished_runs_and_refuses_other_settings(grid_dir, monkeypatch):
    args = ["--out", str(grid_dir), "--device", "cpu"]
    dropout_eval.main([*args, "--passes", "2"])

    def train_model(*args, **kwargs):
        raise AssertionError("a finished run was trained again")

    monkeypatch.setattr(dropout_eval, "train_model", train_model)
    dropout_eval.main([*args, "--passes", "2"])
    with pytest.raises(SystemExit, match="other settings"):
        dropout_eval.main([*args, "--passes", "3"])


def test_table_reports_the_difference_run_by_run():
    runs = [
        {
            "model": "cnn",
            "dropout": 0.2,
            "optimizer": "adamax",
            "seed": seed,
            "test_acc_dropout_off": off,
            "test_acc_dropout_on": on,
        }
        for seed, off, on in [(0, 99.3, [99.25, 99.15]), (1, 99.1, [99.0, 99.0])]
    ]
    row = dropout_eval.dropout_table(runs).splitlines()[2]
    assert row == "| CNN | 0.2 | AdaMax | 99.20 ± 0.14 | 99.10 ± 0.14 | +0.10 ± 0.00 |"
