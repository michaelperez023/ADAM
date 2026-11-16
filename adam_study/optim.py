"""Adam and AdaMax, written from Algorithms 1 and 2 of Kingma & Ba (2015).

Where the paper leaves a detail open, both classes make the same choice as
``torch.optim.Adam`` and ``torch.optim.Adamax``, so they reproduce PyTorch's updates
for the same arguments (``tests/test_optim.py`` checks this):

* Adam adds ``eps`` to the bias-corrected ``sqrt(v_hat)``, as in Algorithm 1.
* AdaMax adds ``eps`` to ``|g_t|`` inside the maximum, which keeps ``u_t`` positive.
* ``weight_decay`` is a coupled L2 penalty: ``weight_decay * theta`` is added to the
  gradient before the moment estimates are updated.

The code structure follows PyTorch's implementation (BSD-3-Clause, see
THIRD_PARTY_NOTICES.md).

Reference: D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization",
ICLR 2015. https://arxiv.org/abs/1412.6980
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import torch
from torch import Tensor
from torch.optim import Optimizer

Params: TypeAlias = Iterable[Tensor] | Iterable[dict[str, Any]]


def _check_hyperparameters(
    lr: float, betas: tuple[float, float], eps: float, weight_decay: float
) -> None:
    if lr < 0.0:
        raise ValueError(f"invalid learning rate: {lr}")
    for i, beta in enumerate(betas):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"invalid beta at index {i}: {beta}")
    if eps < 0.0:
        raise ValueError(f"invalid eps: {eps}")
    if weight_decay < 0.0:
        raise ValueError(f"invalid weight_decay: {weight_decay}")


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015, Algorithm 1).

    For each parameter ``theta`` with stochastic gradient ``g_t`` at update
    ``t = 1, 2, ...``::

        g_t     <- g_t + weight_decay * theta_{t-1}        (optional L2 penalty)
        m_t     <- beta1 * m_{t-1} + (1 - beta1) * g_t     (first moment estimate)
        v_t     <- beta2 * v_{t-1} + (1 - beta2) * g_t^2   (second raw moment estimate)
        m_hat   <- m_t / (1 - beta1^t)                     (bias corrections)
        v_hat   <- v_t / (1 - beta2^t)
        theta_t <- theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    Both moment estimates start at zero, which biases them towards zero during the
    first updates; dividing by ``1 - beta^t`` removes that bias.

    Args:
        params: Parameters to optimize, or dicts that define parameter groups.
        lr: Step size alpha. Learning-rate schedulers update it per parameter group.
        betas: Exponential decay rates (beta1, beta2) of the moment estimates.
        eps: Constant added to ``sqrt(v_hat)`` for numerical stability.
        weight_decay: Coefficient of the coupled L2 penalty; it corresponds to adding
            ``weight_decay / 2 * ||theta||^2`` to the loss.
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        _check_hyperparameters(lr, betas, eps, weight_decay)
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Update every parameter that has a gradient.

        Args:
            closure: Optional function that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2) = group["lr"], group["betas"]
            eps, weight_decay = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                g = p.grad if weight_decay == 0 else p.grad.add(p, alpha=weight_decay)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # lr * m_hat / (sqrt(v_hat) + eps), without materializing m_hat and v_hat.
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias_correction1)

        return loss


class AdaMax(Optimizer):
    """AdaMax, the infinity-norm variant of Adam (Kingma & Ba, 2015, Algorithm 2).

    For each parameter ``theta`` with stochastic gradient ``g_t`` at update
    ``t = 1, 2, ...``::

        g_t     <- g_t + weight_decay * theta_{t-1}      (optional L2 penalty)
        m_t     <- beta1 * m_{t-1} + (1 - beta1) * g_t   (first moment estimate)
        u_t     <- max(beta2 * u_{t-1}, |g_t| + eps)     (weighted infinity norm)
        theta_t <- theta_{t-1} - lr / (1 - beta1^t) * m_t / u_t

    Only ``m_t`` needs a bias correction: ``u_t`` is a running maximum, not an average
    started at zero. The paper's ``u_t`` has no ``eps``; adding it to ``|g_t|`` avoids
    a division by zero when an entry of the gradient has been zero since the first step.

    Args:
        params: Parameters to optimize, or dicts that define parameter groups.
        lr: Step size alpha (the paper suggests 0.002). Learning-rate schedulers update
            it per parameter group.
        betas: Exponential decay rates (beta1, beta2) of the moment estimates.
        eps: Constant added to ``|g_t|`` before taking the maximum.
        weight_decay: Coefficient of the coupled L2 penalty; it corresponds to adding
            ``weight_decay / 2 * ||theta||^2`` to the loss.
    """

    def __init__(
        self,
        params: Params,
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        _check_hyperparameters(lr, betas, eps, weight_decay)
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Update every parameter that has a gradient.

        Args:
            closure: Optional function that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2) = group["lr"], group["betas"]
            eps, weight_decay = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdaMax does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_inf"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]
                m, u = state["exp_avg"], state["exp_inf"]

                g = p.grad if weight_decay == 0 else p.grad.add(p, alpha=weight_decay)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                torch.maximum(u.mul_(beta2), g.abs().add_(eps), out=u)
                p.addcdiv_(m, u, value=-lr / (1 - beta1**t))

        return loss
