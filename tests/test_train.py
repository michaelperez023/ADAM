"""End-to-end training on synthetic data (no MNIST download)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from adam_study.data import make_loader
from adam_study.models import build_model
from adam_study.train import (
    RunConfig,
    evaluate,
    fit,
    make_optimizer,
    make_scheduler,
    train_model,
)


@pytest.mark.parametrize("model", ["logreg", "mlp", "cnn"])
@pytest.mark.parametrize("optimizer", ["adam", "adamax"])
def test_training_reduces_the_loss(digits, model, optimizer):
    config = RunConfig(model=model, optimizer=optimizer, lr=3e-3, epochs=3, batch_size=64)
    history = fit(config, *digits)
    assert len(history.train_loss) == len(history.test_loss) == len(history.test_acc) == 3
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.test_loss[-1] < history.test_loss[0]
    assert history.test_acc[-1] > 90.0


def test_same_seed_reproduces_the_run(digits):
    config = RunConfig(model="mlp", optimizer="adam", dropout=0.2, epochs=2, batch_size=64, seed=7)
    first = fit(config, *digits)
    assert fit(config, *digits) == first
    assert fit(replace(config, seed=8), *digits) != first


def test_train_model_returns_the_trained_model(digits):
    config = RunConfig(model="cnn", optimizer="adamax", dropout=0.2, epochs=2, batch_size=64)
    model, history = train_model(config, *digits)
    assert history == fit(config, *digits)
    test_loss, test_acc = evaluate(model, make_loader(digits[1], 1000), torch.device("cpu"))
    assert (test_loss, test_acc) == (history.test_loss[-1], history.test_acc[-1])


def test_evaluation_disables_dropout(digits):
    torch.manual_seed(0)
    model = build_model("mlp", dropout=0.5)
    loader = make_loader(digits[1], batch_size=64)
    device = torch.device("cpu")
    first = evaluate(model, loader, device)
    assert not model.training
    assert evaluate(model, loader, device) == first


def test_inv_sqrt_schedule_decays_the_step_size_per_update():
    param = torch.zeros(1, requires_grad=True)
    optimizer = make_optimizer("adam", [param], lr=0.1)
    scheduler = make_scheduler(optimizer, "inv_sqrt")
    step_sizes = []
    for _ in range(4):
        step_sizes.append(optimizer.param_groups[0]["lr"])
        param.grad = torch.ones(1)
        optimizer.step()
        scheduler.step()
    assert step_sizes == pytest.approx([0.1 / math.sqrt(t) for t in (1, 2, 3, 4)])


def test_schedule_and_weight_decay_options_train(digits):
    config = RunConfig(
        model="logreg",
        optimizer="adamax",
        schedule="inv_sqrt",
        weight_decay=1e-4,
        lr=1e-2,
        epochs=2,
        batch_size=64,
    )
    history = fit(config, *digits)
    assert all(math.isfinite(v) for v in history.train_loss + history.test_loss)
    assert history.train_loss[-1] < history.train_loss[0]


def test_unknown_names_are_rejected():
    param = torch.zeros(1, requires_grad=True)
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer("sgd", [param], lr=0.1)
    optimizer = make_optimizer("adamax", [param], lr=0.1)
    assert make_scheduler(optimizer, "constant") is None
    with pytest.raises(ValueError, match="unknown schedule"):
        make_scheduler(optimizer, "cosine")
