"""Adam and AdaMax must reproduce torch.optim.Adam and torch.optim.Adamax."""

from __future__ import annotations

import copy
import io
import math
import warnings
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from adam_study.optim import Adam, AdaMax

PAIRS = {"adam": (Adam, torch.optim.Adam), "adamax": (AdaMax, torch.optim.Adamax)}
# float64 shows that the arithmetic is the same; float32 is what training uses.
TOLERANCES = {
    torch.float64: {"rtol": 1e-12, "atol": 1e-12},
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
}


def random_params(generator: torch.Generator, dtype: torch.dtype = torch.float32) -> list[Tensor]:
    shapes = [(7, 5), (11,), (3, 2, 4)]
    return [torch.randn(shape, generator=generator, dtype=dtype) for shape in shapes]


def leaf_copies(tensors: list[Tensor]) -> list[Tensor]:
    return [t.detach().clone().requires_grad_() for t in tensors]


def apply_gradients(optimizer: Optimizer, params: list[Tensor], grads: list[Tensor]) -> None:
    for param, grad in zip(params, grads, strict=True):
        param.grad = grad.clone()
    optimizer.step()


@pytest.mark.parametrize("name", list(PAIRS))
@pytest.mark.parametrize("lr", [1e-3, 1e-2, 1e-1])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_torch_over_many_random_steps(name, lr, weight_decay, dtype):
    ours_cls, torch_cls = PAIRS[name]
    generator = torch.Generator().manual_seed(0)
    initial = random_params(generator, dtype)
    ours, theirs = leaf_copies(initial), leaf_copies(initial)
    ours_opt = ours_cls(ours, lr=lr, weight_decay=weight_decay)
    torch_opt = torch_cls(theirs, lr=lr, weight_decay=weight_decay)

    for _ in range(500):
        grads = [torch.randn(p.shape, generator=generator, dtype=dtype) for p in initial]
        apply_gradients(ours_opt, ours, grads)
        apply_gradients(torch_opt, theirs, grads)

    for p, q in zip(ours, theirs, strict=True):
        torch.testing.assert_close(p, q, **TOLERANCES[dtype])
        state, torch_state = ours_opt.state[p], torch_opt.state[q]
        assert state["step"] == int(torch_state["step"]) == 500
        for key in state.keys() - {"step"}:  # exp_avg and exp_avg_sq or exp_inf
            torch.testing.assert_close(state[key], torch_state[key], **TOLERANCES[dtype])


def make_closure(model: nn.Module, optimizer: Optimizer, x: Tensor, y: Tensor) -> Callable:
    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        return loss

    return closure


@pytest.mark.parametrize("name", list(PAIRS))
def test_matches_torch_when_training_a_model_with_parameter_groups(name):
    ours_cls, torch_cls = PAIRS[name]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 3)).double()
    reference = copy.deepcopy(model)
    x, y = torch.randn(64, 10, dtype=torch.float64), torch.randint(0, 3, (64,))

    def groups(m: nn.Sequential) -> list[dict]:
        return [
            {"params": m[0].parameters(), "lr": 1e-2},
            {"params": m[2].parameters(), "weight_decay": 1e-3},
        ]

    ours_opt, torch_opt = ours_cls(groups(model), lr=1e-3), torch_cls(groups(reference), lr=1e-3)
    ours_step = make_closure(model, ours_opt, x, y)
    torch_step = make_closure(reference, torch_opt, x, y)
    for _ in range(200):
        ours_loss, torch_loss = ours_opt.step(ours_step), torch_opt.step(torch_step)
        torch.testing.assert_close(ours_loss, torch_loss, rtol=1e-10, atol=1e-10)

    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("cls", [Adam, AdaMax])
def test_parameters_without_gradients_are_left_alone(cls):
    used, unused = torch.ones(3, requires_grad=True), torch.ones(3, requires_grad=True)
    optimizer = cls([used, unused], lr=0.1)
    used.grad = torch.ones(3)
    optimizer.step()
    assert not torch.equal(used, torch.ones(3))
    assert torch.equal(unused, torch.ones(3))
    assert unused not in optimizer.state


@pytest.mark.parametrize("cls", [Adam, AdaMax])
def test_state_dict_round_trip_continues_identically(cls):
    generator = torch.Generator().manual_seed(0)
    initial = random_params(generator)
    grads = [[torch.randn(p.shape, generator=generator) for p in initial] for _ in range(20)]
    params = leaf_copies(initial)
    optimizer = cls(params, lr=1e-2, weight_decay=1e-3)
    for step_grads in grads[:10]:
        apply_gradients(optimizer, params, step_grads)

    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    restored = leaf_copies(params)
    restored_optimizer = cls(restored, lr=0.5)  # loading replaces these hyperparameters
    restored_optimizer.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored_optimizer.param_groups[0]["lr"] == 1e-2
    assert restored_optimizer.state[restored[0]]["step"] == 10

    for step_grads in grads[10:]:
        apply_gradients(optimizer, params, step_grads)
        apply_gradients(restored_optimizer, restored, step_grads)
    for p, q in zip(params, restored, strict=True):
        assert torch.equal(p, q)


@pytest.mark.parametrize("name", list(PAIRS))
def test_lambda_lr_schedule_matches_torch(name):
    ours_cls, torch_cls = PAIRS[name]
    generator = torch.Generator().manual_seed(0)
    initial = random_params(generator, torch.float64)
    ours, theirs = leaf_copies(initial), leaf_copies(initial)
    ours_opt, torch_opt = ours_cls(ours, lr=0.1), torch_cls(theirs, lr=0.1)

    def inv_sqrt(done: int) -> float:
        return 1.0 / math.sqrt(done + 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schedulers = [LambdaLR(ours_opt, inv_sqrt), LambdaLR(torch_opt, inv_sqrt)]
        for _ in range(100):
            grads = [torch.randn(p.shape, generator=generator, dtype=p.dtype) for p in initial]
            apply_gradients(ours_opt, ours, grads)
            apply_gradients(torch_opt, theirs, grads)
            for scheduler in schedulers:
                scheduler.step()

    # The scheduler warns if it cannot track optimizer.step() calls on this optimizer.
    assert not [w for w in caught if "step()" in str(w.message)]
    assert ours_opt.param_groups[0]["lr"] == pytest.approx(0.1 / math.sqrt(101))
    for p, q in zip(ours, theirs, strict=True):
        torch.testing.assert_close(p, q, **TOLERANCES[torch.float64])


def test_step_returns_the_closure_loss():
    param = torch.zeros(3, requires_grad=True)
    optimizer = Adam([param])

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = (param - 1).pow(2).sum()
        loss.backward()
        return loss

    assert optimizer.step(closure).item() == 3.0
    assert optimizer.step() is None


@pytest.mark.parametrize("cls", [Adam, AdaMax])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1e-3},
        {"betas": (1.0, 0.999)},
        {"betas": (0.9, -0.1)},
        {"eps": -1e-8},
        {"weight_decay": -0.1},
    ],
)
def test_rejects_invalid_hyperparameters(cls, kwargs):
    with pytest.raises(ValueError, match="invalid"):
        cls([torch.zeros(1, requires_grad=True)], **kwargs)
