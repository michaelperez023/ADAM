"""Model shapes, sizes and dropout behavior."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from adam_study.models import CNN, build_model

CONFIGS = [("logreg", 0.0), ("mlp", 0.0), ("mlp", 0.2), ("cnn", 0.0), ("cnn", 0.2)]
EXPECTED_PARAMETERS = {
    "logreg": 784 * 10 + 10,  # 7,850
    "mlp": (784 * 1000 + 1000) + (1000 * 1000 + 1000) + (1000 * 10 + 10),  # 1,796,010
    "cnn": (25 * 10 + 10) + (10 * 25 * 50 + 50) + (450 * 200 + 200) + (200 * 10 + 10),  # 105,020
}


@pytest.mark.parametrize(("name", "dropout"), CONFIGS)
def test_returns_one_logit_per_class(name, dropout):
    model = build_model(name, dropout)
    assert model(torch.rand(5, 1, 28, 28)).shape == (5, 10)


@pytest.mark.parametrize(("name", "dropout"), CONFIGS)
def test_parameter_count(name, dropout):
    model = build_model(name, dropout)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS[name]


@pytest.mark.parametrize(("name", "dropout"), CONFIGS)
def test_every_parameter_receives_a_gradient(name, dropout):
    torch.manual_seed(0)
    model = build_model(name, dropout)
    F.cross_entropy(model(torch.rand(8, 1, 28, 28)), torch.arange(8)).backward()
    for param_name, param in model.named_parameters():
        assert param.grad is not None, param_name
        assert param.grad.abs().sum() > 0, param_name


def test_cnn_feature_map_shape():
    assert CNN().features(torch.rand(2, 1, 28, 28)).shape == (2, 50, 3, 3)


def test_cnn_returns_logits_not_log_probabilities():
    torch.manual_seed(0)
    logits = CNN()(torch.rand(4, 1, 28, 28))
    # Log-probabilities would have logsumexp == 0 in every row.
    assert not torch.allclose(logits.logsumexp(dim=1), torch.zeros(4))


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_dropout_is_active_in_training_mode_only(name):
    torch.manual_seed(0)
    model = build_model(name, dropout=0.2)
    without_dropout = build_model(name)
    without_dropout.load_state_dict(model.state_dict())
    x = torch.rand(16, 1, 28, 28)

    model.train()
    assert not torch.equal(model(x), model(x))

    model.eval()
    without_dropout.eval()
    assert torch.equal(model(x), model(x))
    torch.testing.assert_close(model(x), without_dropout(x))


@pytest.mark.parametrize("name", ["logreg", "mlp", "cnn"])
def test_without_dropout_training_and_eval_modes_agree(name):
    model = build_model(name)
    x = torch.rand(4, 1, 28, 28)
    model.train()
    out = model(x)
    model.eval()
    torch.testing.assert_close(model(x), out)


def test_build_model_rejects_bad_arguments():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("resnet")
    with pytest.raises(ValueError, match="no dropout"):
        build_model("logreg", dropout=0.2)
    with pytest.raises(ValueError, match="dropout must be"):
        build_model("mlp", dropout=1.0)
