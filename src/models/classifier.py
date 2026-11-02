"""
The referee: a small CNN that recognises the 7 lesion types.

It has three jobs in the project:
1. baseline classifier, trained on real images only, then with the fixes
   against imbalance (weighted loss, oversampling, synthetic images);
2. feature extractor: features() turns an image into a vector, used for the
   "domain FID" and generative precision / recall (question Q2);
3. judge of generated images: does an image generated as "mel" get
   recognised as mel? (conditional fidelity).

Input: images (N, 3, H, W) in [-1, 1]. Works for 28×28 and 64×64.

Quick check: python -m src.models.classifier
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from src import config


def conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    """Two 3×3 convolutions, then halve the image size.

    BatchNorm stabilises training; Dropout2d switches off whole feature maps
    at random, which limits overfitting on a small dataset.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class Classifier(nn.Module):
    """3 convolution blocks -> global average pooling -> linear layer.

    With width=32: 3×28×28 -> 32×14×14 -> 64×7×7 -> 128×3×3 -> vector of 128
    -> 7 class scores. About 0.3 million parameters: small enough to train in
    minutes, and to run on a laptop CPU in the demo.

    Every model of the project stores its constructor arguments in
    self.config, so it can be rebuilt from a saved file:
        model = Classifier(**cfg)
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, width: int = 32,
                 dropout: float = 0.2, head_dropout: float = 0.3, in_channels: int = 3):
        super().__init__()
        self.config = dict(num_classes=num_classes, width=width, dropout=dropout,
                           head_dropout=head_dropout, in_channels=in_channels)
        w = width
        self.body = nn.Sequential(
            conv_block(in_channels, w, dropout),   # 28 -> 14
            conv_block(w, 2 * w, dropout),         # 14 -> 7
            conv_block(2 * w, 4 * w, dropout),     # 7  -> 3
            nn.AdaptiveAvgPool2d(1),               # average over the image: any input size works
            nn.Flatten(),
        )
        self.feature_dim = 4 * w
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(self.feature_dim, num_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> feature vector (N, feature_dim). Used by the domain FID."""
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> class scores (logits), shape (N, num_classes)."""
        return self.head(self.body(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Most likely class index for each image."""
        return self(x).argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probability of each class for each image (rows sum to 1)."""
        return self(x).softmax(dim=1)


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------


@torch.no_grad()
def predict_loader(model: nn.Module, loader, device) -> tuple:
    """Run the model over a DataLoader. Returns (y_true, y_pred, probabilities)."""
    model.eval()
    y_true, probs = [], []
    for xb, yb in loader:
        probs.append(model(xb.to(device)).softmax(dim=1).cpu())
        y_true.append(yb)
    probs = torch.cat(probs).numpy()
    return torch.cat(y_true).numpy(), probs.argmax(axis=1), probs


def classification_metrics(y_true, y_pred) -> dict:
    """The metrics of the project, suited to unbalanced classes.

    accuracy           share of correct answers (misleading here, kept for reference)
    balanced_accuracy  mean of the per-class recalls: each class counts the same
    macro_f1           mean F1 over the classes
    recall_<class>     share of the images of that class that are found
    """
    labels = list(range(config.NUM_CLASSES))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics = {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    metrics.update({f"recall_{c}": float(r) for c, r in zip(config.CLASSES, recalls)})
    return metrics


def evaluate(model: nn.Module, loader, device) -> tuple:
    """Metrics + confusion matrix (rows = true class, columns = predicted class)."""
    y_true, y_pred, _ = predict_loader(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    return classification_metrics(y_true, y_pred), cm


@torch.no_grad()
def extract_features(model: Classifier, images: torch.Tensor, device, batch_size: int = 256) -> np.ndarray:
    """Feature vectors of many images, (N, 3, H, W) in [-1, 1] -> numpy (N, feature_dim).

    Processed in batches so that thousands of images fit in memory.
    """
    model.eval()
    out = [model.features(images[i:i + batch_size].to(device)).cpu()
           for i in range(0, len(images), batch_size)]
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------
# Quick check: python -m src.models.classifier
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from src import utils

    model = Classifier()
    print(f"Parameters: {utils.count_parameters(model):,}".replace(",", " "))
    for size in config.IMG_SIZES:
        x = torch.rand(4, 3, size, size) * 2 - 1
        print(f"{size}×{size}: logits {tuple(model(x).shape)}, features {tuple(model.features(x).shape)}")
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 5, 5, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5])
    m = classification_metrics(y_true, y_pred)
    print(f"Toy metrics: accuracy {m['accuracy']:.2f}, balanced accuracy {m['balanced_accuracy']:.2f}, "
          f"recall_vasc {m['recall_vasc']:.2f}")
