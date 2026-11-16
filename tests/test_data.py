"""The seeded minibatch loader, which must give the same batches as DataLoader."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from adam_study.data import make_loader, to_device


@pytest.fixture
def dataset() -> TensorDataset:
    return TensorDataset(torch.arange(1000.0).reshape(500, 2), torch.arange(500))


@pytest.mark.parametrize("shuffle", [True, False])
def test_same_batches_as_dataloader(dataset, shuffle):
    ours = make_loader(dataset, batch_size=64, shuffle=shuffle, seed=3)
    reference = DataLoader(
        dataset, batch_size=64, shuffle=shuffle, generator=torch.Generator().manual_seed(3)
    )
    assert len(ours) == len(reference) == 8
    for _ in range(3):  # reshuffled every epoch, identically
        for (x, y), (ref_x, ref_y) in zip(ours, reference, strict=True):
            assert torch.equal(x, ref_x)
            assert torch.equal(y, ref_y)


def test_shuffling_depends_only_on_the_seed(dataset):
    def epoch(seed: int) -> list[int]:
        return torch.cat([y for _, y in make_loader(dataset, 50, shuffle=True, seed=seed)]).tolist()

    first = epoch(0)
    torch.manual_seed(123)  # the global RNG must not matter
    assert epoch(0) == first
    assert epoch(1) != first
    assert sorted(first) == list(range(len(dataset)))


def test_iterating_leaves_the_global_rng_untouched(dataset):
    torch.manual_seed(0)
    state = torch.get_rng_state()
    for _ in make_loader(dataset, 64, shuffle=True, seed=1):
        pass
    assert torch.equal(torch.get_rng_state(), state)


def test_to_device_moves_every_tensor(dataset):
    moved = to_device(dataset, torch.device("meta"))
    assert [t.device.type for t in moved.tensors] == ["meta", "meta"]
    assert [t.shape for t in moved.tensors] == [t.shape for t in dataset.tensors]
