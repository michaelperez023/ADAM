"""Training and evaluation loop shared by every experiment."""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset

from adam_study.data import make_loader
from adam_study.models import build_model
from adam_study.optim import Adam, AdaMax

log = logging.getLogger(__name__)

OPTIMIZERS: dict[str, type[Optimizer]] = {"adam": Adam, "adamax": AdaMax}
OPTIMIZER_LABELS = {"adam": "Adam", "adamax": "AdaMax"}
SCHEDULES = ("constant", "inv_sqrt")


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines one training run."""

    model: str = "logreg"
    optimizer: str = "adam"
    dropout: float = 0.0
    seed: int = 0
    lr: float = 1e-3
    schedule: str = "constant"
    weight_decay: float = 0.0
    epochs: int = 20
    batch_size: int = 128

    @property
    def label(self) -> str:
        dropout = f" dropout={self.dropout:g}" if self.dropout else ""
        return f"{self.model}{dropout} {self.optimizer} seed={self.seed}"


@dataclass
class History:
    """Per-epoch metrics of one run; accuracy is in percent."""

    train_loss: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, and ask cuDNN for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - the global RNG is what libraries draw from
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> torch.device:
    """Map ``"auto"`` to CUDA when it is available and to the CPU otherwise.

    Other backends such as ``"mps"`` are used only when requested by name.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_optimizer(
    name: str, params: Iterable[Tensor], lr: float, weight_decay: float = 0.0
) -> Optimizer:
    """Create ``"adam"`` or ``"adamax"`` from :mod:`adam_study.optim`."""
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; expected one of {', '.join(OPTIMIZERS)}")
    return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer: Optimizer, schedule: str) -> LRScheduler | None:
    """Return the step-size schedule, stepped after every update, or None for a constant one.

    ``"inv_sqrt"`` is the decay used in the paper's analysis and logistic regression
    experiment: alpha_t = alpha / sqrt(t) for updates t = 1, 2, ...
    """
    if schedule == "constant":
        return None
    if schedule == "inv_sqrt":
        # LambdaLR passes the number of updates already made, which is t - 1.
        return LambdaLR(optimizer, lambda done: 1.0 / math.sqrt(done + 1))
    raise ValueError(f"unknown schedule {schedule!r}; expected one of {', '.join(SCHEDULES)}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    scheduler: LRScheduler | None = None,
) -> float:
    """Make one pass over ``loader`` and return the mean training loss per example."""
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (mean loss, accuracy in percent), with dropout disabled."""
    model.eval()
    total, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += len(y)
    return total / count, 100.0 * correct / count


def train_model(
    config: RunConfig,
    train_set: TensorDataset,
    test_set: TensorDataset,
    device: torch.device | str = "cpu",
    test_batch_size: int = 1000,
) -> tuple[nn.Module, History]:
    """Train a fresh model as described by ``config``; return it and its per-epoch metrics.

    Seeding comes first, so runs that differ only in the optimizer start from the same
    weights and see the same minibatches and dropout masks.
    """
    device = torch.device(device)
    seed_everything(config.seed)
    model = build_model(config.model, config.dropout).to(device)
    optimizer = make_optimizer(config.optimizer, model.parameters(), config.lr, config.weight_decay)
    scheduler = make_scheduler(optimizer, config.schedule)
    train_loader = make_loader(train_set, config.batch_size, shuffle=True, seed=config.seed)
    test_loader = make_loader(test_set, test_batch_size)

    history = History()
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        test_loss, test_acc = evaluate(model, test_loader, device)
        history.train_loss.append(train_loss)
        history.test_loss.append(test_loss)
        history.test_acc.append(test_acc)
        log.info(
            "%s | epoch %d/%d | train loss %.4f | test loss %.4f | test acc %.2f%%",
            config.label,
            epoch,
            config.epochs,
            train_loss,
            test_loss,
            test_acc,
        )
    return model, history


def fit(
    config: RunConfig,
    train_set: TensorDataset,
    test_set: TensorDataset,
    device: torch.device | str = "cpu",
    test_batch_size: int = 1000,
) -> History:
    """Train a fresh model as described by ``config`` and return its per-epoch metrics."""
    return train_model(config, train_set, test_set, device, test_batch_size)[1]
