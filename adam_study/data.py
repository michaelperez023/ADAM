"""MNIST as in-memory tensors, and a minibatch loader with seeded shuffling."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torchvision import datasets


def load_mnist(
    root: str | Path = "data", download: bool = True
) -> tuple[TensorDataset, TensorDataset]:
    """Return the MNIST training set (60,000 images) and test set (10,000 images).

    Images are float32 tensors of shape (N, 1, 28, 28) scaled to [0, 1], the same values
    ``torchvision.transforms.ToTensor`` produces; labels are int64. The raw files are
    downloaded to ``root`` on first use (about 12 MB).
    """
    splits = []
    for train in (True, False):
        mnist = datasets.MNIST(root, train=train, download=download)
        images = mnist.data.unsqueeze(1).float().div_(255)
        splits.append(TensorDataset(images, mnist.targets))
    train_set, test_set = splits
    return train_set, test_set


def to_device(dataset: TensorDataset, device: torch.device) -> TensorDataset:
    """Return a dataset with the tensors of ``dataset`` moved to ``device``."""
    return TensorDataset(*(tensor.to(device) for tensor in dataset.tensors))


def make_loader(
    dataset: TensorDataset, batch_size: int, shuffle: bool = False, seed: int = 0
) -> DataLoader:
    """Return a minibatch loader whose shuffling depends only on ``seed``.

    It yields the same batches as ``DataLoader(dataset, batch_size, shuffle=shuffle,
    generator=torch.Generator().manual_seed(seed))`` and reshuffles every epoch, but it
    gathers each minibatch with one indexing call instead of one call per example, which
    is much faster for tensors that are already in memory.
    """
    generator = torch.Generator().manual_seed(seed)
    order = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    batches = BatchSampler(order, batch_size, drop_last=False)
    # With batch_size=None every item the sampler yields (a list of indices) is one batch.
    return DataLoader(dataset, sampler=batches, batch_size=None, generator=generator)
