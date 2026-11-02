"""
Conditional Variational Autoencoder (cVAE): the baseline generator.

Idea in three lines:
- the ENCODER q(z | x, y) summarises an image x of class y as a small vector z;
- the DECODER p(x | z, y) rebuilds an image from z and the requested class y;
- to GENERATE, draw z ~ N(0, I) and give it to the decoder with any class.

Training minimises the negative ELBO:
    loss = ||x - x_hat||^2  +  beta * KL( q(z | x, y) || N(0, I) )
           reconstruction       keeps the latent space close to N(0, I),
                                so that random z decode into plausible images

Same interface as every generator of the project:
    images = model.sample(labels)   # labels: class indices -> images in [-1, 1]

Works for 28×28 and 64×64 images. Quick check: python -m src.models.cvae
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


def _down(c_in: int, c_out: int) -> nn.Sequential:
    """Halve the image size: 4×4 convolution with stride 2."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4, 2, 1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))


def _up(c_in: int, c_out: int) -> nn.Sequential:
    """Double the image size: 4×4 transposed convolution with stride 2."""
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))


class CVAE(nn.Module):
    """Convolutional cVAE.

    28×28: 3×28×28 -> 32×14×14 -> 64×7×7 -> vector -> (mu, logvar) of size latent_dim
    64×64: 3×64×64 -> 32×32×32 -> 64×16×16 -> 128×8×8 -> vector -> (mu, logvar)
    The decoder mirrors the encoder. The class y enters both networks as a
    one-hot vector: this is what makes the model conditional.
    """

    def __init__(self, latent_dim: int = 16, num_classes: int = config.NUM_CLASSES,
                 img_size: int = config.IMG_SIZE, base_channels: int = 32, hidden: int = 256,
                 in_channels: int = 3):
        super().__init__()
        if img_size not in (28, 64):
            raise ValueError("img_size must be 28 or 64")
        self.config = dict(latent_dim=latent_dim, num_classes=num_classes, img_size=img_size,
                           base_channels=base_channels, hidden=hidden, in_channels=in_channels)
        self.latent_dim, self.num_classes = latent_dim, num_classes

        n_down = 2 if img_size == 28 else 3
        channels = [base_channels * 2 ** i for i in range(n_down)]   # 32, 64 (, 128)
        self.s = img_size // 2 ** n_down                              # 7 or 8
        self.c_last = channels[-1]
        flat = self.c_last * self.s * self.s

        # ---- Encoder
        layers, c = [], in_channels
        for c_out in channels:
            layers.append(_down(c, c_out))
            c = c_out
        self.enc_conv = nn.Sequential(*layers, nn.Flatten())
        self.enc_fc = nn.Sequential(nn.Linear(flat + num_classes, hidden), nn.LeakyReLU(0.2))
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        # ---- Decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, flat), nn.LeakyReLU(0.2),
        )
        layers = []
        for c_in, c_out in zip(channels[::-1], channels[::-1][1:] + [base_channels]):
            layers.append(_up(c_in, c_out))
        # A last 3×3 convolution smooths the "checkerboard" artefacts of
        # transposed convolutions; tanh gives values in [-1, 1].
        self.dec_conv = nn.Sequential(*layers, nn.Conv2d(base_channels, in_channels, 3, 1, 1), nn.Tanh())

    # ------------------------------------------------------------------ parts

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, self.num_classes).float()

    def encode(self, x: torch.Tensor, y: torch.Tensor):
        """q(z | x, y): returns the mean and log-variance of z."""
        h = self.enc_fc(torch.cat([self.enc_conv(x), self.one_hot(y)], dim=1))
        logvar = self.fc_logvar(h).clamp(-10, 10)   # avoids exp() overflow early in training
        return self.fc_mu(h), logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I).

        The randomness is in eps, so gradients can flow through mu and sigma:
        this is the reparameterization trick.
        """
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """p(x | z, y): image in [-1, 1] from a latent vector and a class."""
        h = self.dec_fc(torch.cat([z, self.one_hot(y)], dim=1))
        return self.dec_conv(h.view(-1, self.c_last, self.s, self.s))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(x, y)
        return self.decode(self.reparameterize(mu, logvar), y), mu, logvar

    # ------------------------------------------------------------------ loss

    def loss(self, x: torch.Tensor, y: torch.Tensor, beta: float = 1.0) -> dict:
        """Negative ELBO, averaged per image.

        rec: sum of squared errors over the pixels (Gaussian likelihood with a
             fixed variance, up to a constant);
        kl:  closed-form KL divergence between N(mu, sigma²) and N(0, I).
        beta > 1 favours a smoother latent space (blurrier images),
        beta < 1 favours sharper reconstructions.
        """
        x_hat, mu, logvar = self(x, y)
        n = len(x)
        rec = F.mse_loss(x_hat, x, reduction="sum") / n
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
        return {"loss": rec + beta * kl, "rec": rec, "kl": kl}

    # ------------------------------------------------------------------ use

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Generate one image per label. Pass a seeded torch.Generator (on CPU)
        to get the same images every time."""
        was_training = self.training
        self.eval()
        z = torch.randn(len(labels), self.latent_dim, generator=generator).to(labels.device)
        images = self.decode(z, labels)
        self.train(was_training)
        return images

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Encode then decode, using the mean of z (no random draw)."""
        was_training = self.training
        self.eval()
        mu, _ = self.encode(x, y)
        images = self.decode(mu, y)
        self.train(was_training)
        return images


# --------------------------------------------------------------------------
# Quick check: python -m src.models.cvae
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from src import utils

    for size in config.IMG_SIZES:
        model = CVAE(img_size=size)
        x = torch.rand(8, 3, size, size) * 2 - 1
        y = torch.randint(0, config.NUM_CLASSES, (8,))
        out = model.loss(x, y)
        gen = model.sample(torch.arange(config.NUM_CLASSES), torch.Generator().manual_seed(0))
        print(f"{size}×{size}: {utils.count_parameters(model):,} parameters".replace(",", " ")
              + f", loss {out['loss'].item():.1f} (rec {out['rec'].item():.1f}, kl {out['kl'].item():.2f})"
              + f", sample {tuple(gen.shape)} in [{gen.min():.2f}, {gen.max():.2f}]")
