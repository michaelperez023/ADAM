"""The three MNIST classifiers compared in the study. Every model returns logits."""

from __future__ import annotations

from torch import Tensor, nn

IMAGE_SIZE = 28
NUM_CLASSES = 10

MODEL_NAMES = ("logreg", "mlp", "cnn")
MODEL_LABELS = {"logreg": "Logistic regression", "mlp": "MLP", "cnn": "CNN"}


def _dropout(p: float) -> nn.Module:
    return nn.Dropout(p) if p > 0 else nn.Identity()


class LogisticRegression(nn.Module):
    """Multinomial logistic regression: one linear layer on the 784 pixel values."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.flatten(x))


class MLP(nn.Module):
    """Fully connected 784-1000-1000-10 network with ReLU activations.

    Dropout, when enabled, is applied once, between the second hidden layer and the
    output layer.
    """

    def __init__(self, hidden: int = 1000, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            _dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class CNN(nn.Module):
    """Two [5x5 convolution -> 3x3 max-pool with stride 2 -> ReLU] stages, then FC 200 -> 10.

    Feature map shapes for one image: 1x28x28 -> 10x24x24 -> 10x11x11 -> 50x7x7 -> 50x3x3.
    Dropout, when enabled, is applied after the 200-unit hidden layer.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(10, 50, kernel_size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 3 * 3, 200),
            nn.ReLU(),
            _dropout(dropout),
            nn.Linear(200, NUM_CLASSES),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


def build_model(name: str, dropout: float = 0.0) -> nn.Module:
    """Create a model by name: ``"logreg"``, ``"mlp"`` or ``"cnn"``."""
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    if name == "logreg":
        if dropout:
            raise ValueError("logistic regression has no dropout layer")
        return LogisticRegression()
    if name == "mlp":
        return MLP(dropout=dropout)
    if name == "cnn":
        return CNN(dropout=dropout)
    raise ValueError(f"unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}")
