"""
Measures of generated images. Plain functions: evaluate.py calls them.

Most metrics compare two sets of FEATURE VECTORS (one per image) rather than
raw pixels. Two feature extractors are available:
- inception_features(): InceptionV3 trained on ImageNet photos (the standard
  FID, comparable with papers, but designed for photos, not 28×28 lesions);
- domain_features(): our referee CNN (weights/classifier.pt), trained on
  lesions. Comparing the two answers question Q2.

Metrics:
    fid()                  Fréchet distance between two feature sets (lower = closer)
    kid()                  Kernel Inception Distance: unbiased, with a std (lower = closer)
    precision_recall()     quality (precision) and coverage (recall), measured separately
    conditional_fidelity() share of images generated as class c that the referee recognises as c
    memorization_report()  are generated images copies of training images?
    classification_accuracy_score()  CAS: train a classifier on generated images only,
                                     test it on real images

Self-test on toy data: python -m src.metrics
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

from src import config, utils

# --------------------------------------------------------------------------
# Feature extractors
# --------------------------------------------------------------------------

_INCEPTION = None


def _get_inception(device):
    """InceptionV3 from torchmetrics / torch-fidelity, loaded once.

    The first call downloads the weights (about 100 MB): it needs internet.
    """
    global _INCEPTION
    if _INCEPTION is None:
        try:
            from torchmetrics.image.fid import NoTrainInceptionV3
        except ImportError as err:
            raise ImportError("Inception features need: pip install 'torchmetrics[image]'") from err
        _INCEPTION = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"])
    return _INCEPTION.to(device).eval()


@torch.no_grad()
def inception_features(images: np.ndarray, device, batch_size: int = 200) -> np.ndarray:
    """uint8 images (N, H, W, 3) -> InceptionV3 features (N, 2048).

    The network resizes the images to 299×299 itself. Our 28×28 lesions are
    enlarged about 10 times: one reason why this FID is debatable here.
    """
    net = _get_inception(device)
    feats = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[i:i + batch_size])).permute(0, 3, 1, 2).to(device)
        out = net(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        feats.append(out.reshape(len(x), -1).double().cpu())
    return torch.cat(feats).numpy()


def load_referee(device):
    """The referee CNN exported by: python -m src.train_classifier ... --export"""
    from src.models.classifier import Classifier

    path = config.WEIGHTS_DIR / "classifier.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Export the referee first: "
                                "python -m src.train_classifier --weighted --augment --seed 0 --export")
    state_dict, cfg = utils.load_weights(path)
    model = Classifier(**cfg)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def domain_features(images: np.ndarray, referee: nn.Module, device, batch_size: int = 256) -> np.ndarray:
    """uint8 images (N, H, W, 3) -> referee features (N, 128), learnt on lesions."""
    from src.models.classifier import extract_features

    return extract_features(referee, utils.uint8_to_tensor(images), device, batch_size).astype(np.float64)


# --------------------------------------------------------------------------
# FID and KID
# --------------------------------------------------------------------------


def fid(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """Fréchet distance between two Gaussians fitted to the feature sets:

        FID = ||mu_a - mu_b||²  +  Tr( S_a + S_b - 2 (S_a S_b)^(1/2) )

    0 means identical mean and covariance. The value depends on the number
    of images (it is biased upwards for small sets), so always compare
    models with the same number of images.
    """
    a, b = np.asarray(feats_a, np.float64), np.asarray(feats_b, np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s_a @ s_b)
    if not np.isfinite(covmean).all():
        # Nearly singular covariances: add a tiny value on the diagonal
        eps = np.eye(len(s_a)) * 1e-6
        covmean = linalg.sqrtm((s_a + eps) @ (s_b + eps))
    covmean = np.real(covmean)  # numerical noise can add tiny imaginary parts
    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(s_a) + np.trace(s_b) - 2 * np.trace(covmean))


def kid(feats_a: np.ndarray, feats_b: np.ndarray, n_subsets: int = 100, subset_size: int = 1000,
        seed: int = 0) -> tuple:
    """Kernel Inception Distance (Bińkowski et al., 2018): squared MMD with the
    polynomial kernel k(x, y) = (x·y / d + 1)³, averaged over random subsets.

    Unlike the FID it is unbiased, and the subsets give a standard deviation
    for free. Returns (mean, std).
    """
    a, b = np.asarray(feats_a, np.float64), np.asarray(feats_b, np.float64)
    d = a.shape[1]
    m = min(subset_size, len(a), len(b))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
               + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        values.append(mmd)
    return float(np.mean(values)), float(np.std(values))


# --------------------------------------------------------------------------
# Precision and recall (Kynkäänniemi et al., 2019)
# --------------------------------------------------------------------------


def _kth_radius(x: torch.Tensor, k: int, batch: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest neighbour in the same set."""
    radii = []
    for i in range(0, len(x), batch):
        d = torch.cdist(x[i:i + batch], x)
        radii.append(d.kthvalue(k + 1, dim=1).values)   # k + 1: the nearest is the point itself
    return torch.cat(radii)


def _covered(points: torch.Tensor, centres: torch.Tensor, radii: torch.Tensor, batch: int) -> torch.Tensor:
    """For each point: is it inside at least one ball (centre, radius)?"""
    inside = []
    for i in range(0, len(points), batch):
        d = torch.cdist(points[i:i + batch], centres)
        inside.append((d <= radii[None, :]).any(dim=1))
    return torch.cat(inside)


def precision_recall(real: np.ndarray, gen: np.ndarray, k: int = 3, batch: int = 1000) -> tuple:
    """Improved precision and recall.

    Each real image gets a ball whose radius is the distance to its k-th
    nearest real neighbour; the union of the balls approximates the
    "manifold" of real images. Same for the generated images.
    precision = share of generated images inside the real manifold  (are they realistic?)
    recall    = share of real images inside the generated manifold  (is all the variety covered?)
    A generator in mode collapse: high precision, low recall.
    """
    r = torch.as_tensor(np.asarray(real), dtype=torch.float32)
    g = torch.as_tensor(np.asarray(gen), dtype=torch.float32)
    precision = _covered(g, r, _kth_radius(r, k, batch), batch).float().mean().item()
    recall = _covered(r, g, _kth_radius(g, k, batch), batch).float().mean().item()
    return precision, recall


# --------------------------------------------------------------------------
# Conditional fidelity
# --------------------------------------------------------------------------


@torch.no_grad()
def conditional_fidelity(referee: nn.Module, images: np.ndarray, labels: np.ndarray, device,
                         batch_size: int = 512) -> dict:
    """Share of images generated as class c that the referee classifies as c.

    Checks that the conditioning works. It is bounded by the referee's own
    recall on real images: compare fidelity_c with the referee's test recall
    for class c rather than with 100 %.
    """
    x = utils.uint8_to_tensor(images)
    preds = torch.cat([referee(x[i:i + batch_size].to(device)).argmax(1).cpu()
                       for i in range(0, len(x), batch_size)]).numpy()
    labels = np.asarray(labels)
    out = {"fidelity": float(np.mean(preds == labels))}
    for c, name in enumerate(config.CLASSES):
        mask = labels == c
        out[f"fidelity_{name}"] = float(np.mean(preds[mask] == c)) if mask.any() else float("nan")
    return out


# --------------------------------------------------------------------------
# Memorisation check
# --------------------------------------------------------------------------


def nearest_neighbours(query: np.ndarray, reference: np.ndarray, batch: int = 500) -> tuple:
    """For each query image, the pixel distance to the closest reference image
    and its index. Images are uint8 (N, H, W, 3), compared in [0, 1]."""
    q = torch.as_tensor(np.asarray(query).reshape(len(query), -1), dtype=torch.float32) / 255
    r = torch.as_tensor(np.asarray(reference).reshape(len(reference), -1), dtype=torch.float32) / 255
    dist, idx = [], []
    for i in range(0, len(q), batch):
        d = torch.cdist(q[i:i + batch], r)
        v, j = d.min(dim=1)
        dist.append(v)
        idx.append(j)
    return torch.cat(dist).numpy(), torch.cat(idx).numpy()


def memorization_report(gen: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    """Are generated images closer to the training images than new real images are?

    Reference: a real test image is, by construction, NOT a copy of a train
    image. If generated images are much closer to their nearest train image
    than test images are, the generator is probably copying.
    nn_ratio < 1: closer than real new images; well below 1 is a warning sign.
    share_suspicious: share of generated images closer to the train set than
    95 % of the test images.
    """
    d_gen, _ = nearest_neighbours(gen, train)
    d_test, _ = nearest_neighbours(test, train)
    threshold = np.percentile(d_test, 5)
    return {
        "nn_gen_median": float(np.median(d_gen)),
        "nn_test_median": float(np.median(d_test)),
        "nn_ratio": float(np.median(d_gen) / np.median(d_test)),
        "share_suspicious": float(np.mean(d_gen < threshold)),
    }


# --------------------------------------------------------------------------
# CAS: Classification Accuracy Score (Ravuri & Vinyals, 2019)
# --------------------------------------------------------------------------


def classification_accuracy_score(images: np.ndarray, labels: np.ndarray, device,
                                  size: int = config.IMG_SIZE, epochs: int = 20, seed: int = 0,
                                  batch_size: int = 128) -> dict:
    """Train the referee architecture on GENERATED images only, then test it on
    REAL test images.

    If the generated images carry the useful information of each class, the
    classifier works on real lesions. CAS penalises mode collapse, which FID
    can miss. The best epoch is chosen on the real val split; the test split
    is used once, at the end.
    """
    from torch.utils.data import DataLoader

    from src.data import HAMDataset, class_weights, get_loader
    from src.models.classifier import Classifier, evaluate

    utils.set_seed(seed)
    train_loader = DataLoader(HAMDataset(images, labels), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = get_loader("val", size=size, batch_size=512)
    test_loader = get_loader("test", size=size, batch_size=512)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        score = evaluate(model, val_loader, device)[0]["balanced_accuracy"]
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)[0]
    return {"cas_balanced_accuracy": test["balanced_accuracy"], "cas_macro_f1": test["macro_f1"],
            "cas_accuracy": test["accuracy"]}


# --------------------------------------------------------------------------
# Self-test on toy data: python -m src.metrics
# --------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (2000, 16))
    same = rng.normal(0, 1, (2000, 16))
    shifted = rng.normal(0.5, 1, (2000, 16))
    collapsed = real[rng.choice(2000, 2000)] * 0.3          # realistic but much less varied

    print("FID   same distribution   :", round(fid(real, same), 3))
    print("FID   shifted by 0.5      :", round(fid(real, shifted), 3))
    print("KID   same / shifted      :", [round(kid(real, x, n_subsets=20)[0], 4) for x in (same, shifted)])
    for name, gen in (("same", same), ("shifted", shifted), ("collapsed", collapsed)):
        p, r = precision_recall(real, gen)
        print(f"P/R   {name:<20}: precision {p:.2f}, recall {r:.2f}")

    imgs = rng.integers(0, 256, (300, 28, 28, 3), dtype=np.uint8)
    copies = imgs[:100].copy()
    new = rng.integers(0, 256, (100, 28, 28, 3), dtype=np.uint8)
    print("Memorisation, exact copies:", {k: round(v, 3) for k, v in memorization_report(copies, imgs, new).items()})
