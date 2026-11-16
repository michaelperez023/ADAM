"""Shared fixtures: a small synthetic stand-in for MNIST, so no test downloads data."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset


def synthetic_digits(n: int, seed: int) -> TensorDataset:
    """28x28 noise images in which the class sets the position of a bright 7x7 square."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 10
    images = 0.3 * torch.rand(n, 1, 28, 28, generator=generator)
    for digit in range(10):
        row, col = divmod(digit, 4)
        images[labels == digit, :, 7 * row : 7 * row + 7, 7 * col : 7 * col + 7] += 0.7
    return TensorDataset(images, labels)


@pytest.fixture(scope="session")
def digits() -> tuple[TensorDataset, TensorDataset]:
    """Synthetic (train, test) sets with 512 and 256 images."""
    return synthetic_digits(512, seed=0), synthetic_digits(256, seed=1)
