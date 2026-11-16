"""The experiment CLI on synthetic data: grid, resuming, summaries and figures."""

from __future__ import annotations

import csv
import json

import pytest

from adam_study import experiments, train


@pytest.fixture
def fake_mnist(monkeypatch, digits):
    monkeypatch.setattr(experiments, "load_mnist", lambda *args, **kwargs: digits)


def count_fit_calls(monkeypatch, fail_after: int | None = None) -> list[str]:
    """Wrap experiments.fit to record each run, optionally interrupting after N runs."""
    calls: list[str] = []

    def fit(config, *args, **kwargs):
        if fail_after is not None and len(calls) == fail_after:
            raise KeyboardInterrupt
        calls.append(config.label)
        return train.fit(config, *args, **kwargs)

    monkeypatch.setattr(experiments, "fit", fit)
    return calls


def test_default_grid_has_the_ten_configurations():
    configs = experiments.grid(["logreg", "mlp", "cnn"], ["adam", "adamax"], [0.0, 0.2])
    assert len(configs) == len(set(configs)) == 10
    assert ("logreg", "adam", 0.0) in configs
    assert ("logreg", "adam", 0.2) not in configs
    assert ("cnn", "adamax", 0.2) in configs


def test_grid_ignores_repeated_choices():
    configs = experiments.grid(["mlp", "mlp"], ["adam", "adam"], [0.0, 0, 0.2])
    assert configs == [("mlp", "adam", 0.0), ("mlp", "adam", 0.2)]


def test_full_grid_writes_results_summaries_and_figures(tmp_path, fake_mnist, monkeypatch):
    args = ["--seeds", "2", "--epochs", "2", "--batch-size", "64", "--out", str(tmp_path)]
    experiments.main([*args, "--device", "cpu"])

    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["settings"] == {
        "epochs": 2,
        "batch_size": 64,
        "lr": 1e-3,
        "schedule": "constant",
        "weight_decay": 0.0,
    }
    assert len(saved["runs"]) == 20
    assert all(len(run["test_acc"]) == 2 for run in saved["runs"])

    with (tmp_path / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [(row["model"], row["dropout"], row["optimizer"]) for row in rows[:4]] == [
        ("logreg", "0", "adam"),
        ("logreg", "0", "adamax"),
        ("mlp", "0", "adam"),
        ("mlp", "0", "adamax"),
    ]
    assert len(rows) == 10
    assert all(row["seeds"] == "2" for row in rows)
    assert "| Logistic regression | none |" in (tmp_path / "summary.md").read_text()

    figures = {path.name for path in (tmp_path / "figures").glob("*.png")}
    assert figures == {
        "summary.png",
        "dropout.png",
        "adam_vs_adamax_logreg.png",
        "adam_vs_adamax_mlp.png",
        "adam_vs_adamax_cnn.png",
    }

    # Everything requested is already saved, so a second invocation trains nothing.
    calls = count_fit_calls(monkeypatch)
    experiments.main([*args, "--no-plots"])
    assert calls == []


def test_interrupted_run_resumes_where_it_stopped(tmp_path, fake_mnist, monkeypatch):
    args = ["--models", "logreg", "--seeds", "3", "--epochs", "1", "--out", str(tmp_path)]
    args += ["--device", "cpu", "--no-plots"]
    count_fit_calls(monkeypatch, fail_after=4)
    with pytest.raises(KeyboardInterrupt):
        experiments.main(args)
    assert len(json.loads((tmp_path / "results.json").read_text())["runs"]) == 4

    calls = count_fit_calls(monkeypatch)
    experiments.main(args)
    assert calls == ["logreg adam seed=2", "logreg adamax seed=2"]
    runs = json.loads((tmp_path / "results.json").read_text())["runs"]
    assert sorted((r["optimizer"], r["seed"]) for r in runs) == [
        ("adam", 0),
        ("adam", 1),
        ("adam", 2),
        ("adamax", 0),
        ("adamax", 1),
        ("adamax", 2),
    ]


@pytest.mark.parametrize(
    "option", [["--epochs", "0"], ["--batch-size", "0"], ["--dropout", "0", "1.0"]]
)
def test_rejects_invalid_options_before_training(tmp_path, fake_mnist, option):
    with pytest.raises(SystemExit):
        experiments.main(["--models", "mlp", "--out", str(tmp_path), "--no-plots", *option])
    assert not (tmp_path / "results.json").exists()


def test_refuses_to_mix_runs_from_different_settings(tmp_path, fake_mnist):
    args = ["--models", "logreg", "--seeds", "1", "--out", str(tmp_path), "--no-plots"]
    experiments.main([*args, "--epochs", "1", "--device", "cpu"])
    with pytest.raises(SystemExit, match="another --out directory"):
        experiments.main([*args, "--epochs", "2", "--device", "cpu"])


def logreg_run(optimizer: str, seed: int, acc: float) -> dict:
    return {
        "model": "logreg",
        "dropout": 0.0,
        "optimizer": optimizer,
        "seed": seed,
        "test_acc": [50.0, acc],
    }


def test_accuracy_table_reports_mean_std_and_paired_difference():
    runs = [
        logreg_run("adam", 0, 92.0),
        logreg_run("adam", 1, 92.2),
        logreg_run("adamax", 0, 91.9),
        logreg_run("adamax", 1, 92.0),
    ]
    table = experiments.accuracy_table(runs).splitlines()
    assert table[0] == "| Model | Dropout | Adam | AdaMax | Adam - AdaMax |"
    assert table[2] == "| Logistic regression | none | 92.10 ± 0.14 | 91.95 ± 0.07 | +0.15 ± 0.07 |"


def test_a_difference_that_rounds_to_zero_has_no_sign():
    runs = [
        logreg_run("adam", 0, 99.0),
        logreg_run("adam", 1, 99.2),
        logreg_run("adamax", 0, 99.004),
        logreg_run("adamax", 1, 99.2),
    ]
    row = experiments.accuracy_table(runs).splitlines()[2]
    assert row.endswith("| 99.10 ± 0.14 | 99.10 ± 0.14 | 0.00 ± 0.00 |")
