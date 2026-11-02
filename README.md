# DermaGen

**Conditional generation of skin lesion images with a cVAE, a GAN and a diffusion model, and the value of synthetic data for classifying rare classes.**

Generative AI module project · Université A. Mira de Béjaïa, Département d'Informatique
Author: Sid Ali Alili

---

## The project in one paragraph

HAM10000 contains 10,015 dermatoscopic images of 7 lesion types, but the classes are very unbalanced: 6,705 melanocytic nevi against only 115 dermatofibromas. We train three families of **conditional** generative models (give them a lesion type, they produce an image of that type): a conditional VAE, a conditional DCGAN and a DDPM with classifier-free guidance. We compare them under identical conditions. Then we use the synthetic images to fill the rare classes and measure whether a classifier recognises them better than with simple fixes (weighted loss, flips and rotations). A Streamlit demo shows how each model generates.

## Research questions

| | Question |
|---|---|
| **Q1** | At a comparable compute budget, which family (cVAE, GAN, diffusion) produces the most realistic and varied lesions, and at what training and sampling cost? |
| **Q2** | Does the standard FID (ImageNet features) rank the models the same way as domain-adapted metrics (FID with our own classifier's features, CAS)? |
| **Q3** | Do synthetic images improve recognition of the rare classes more than weighted loss, oversampling or classic augmentation? |

## Dataset

**Skin Cancer MNIST: HAM10000** on Kaggle: <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>

Put the download in `data/raw/` like this:

```
data/raw/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv
```

The `hmnist_*.csv` files are not needed. They contain no `lesion_id`, so they cannot be split cleanly.

**Important:** one lesion can have several photos. We split train / val / test **by `lesion_id`**, so that no lesion appears in two splits. A random split would leak information and inflate the scores.

Licence: non-commercial use only. Source: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018).

## Project structure

```
dermagen/
├── data/          raw download (raw/) and prepared arrays (processed/), not in Git
├── src/           all the code: data preparation, models, training, evaluation
│   └── models/    classifier, cVAE, cGAN, U-Net, DDPM
├── notebooks/     exploration, Kaggle training, report figures
├── weights/       final trained models (fp16)
├── results/       training logs, generated samples, figures, metrics tables
├── app/           Streamlit demo
└── report/        report and slides
```

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

On Kaggle, PyTorch is already installed; the notebook installs the rest.

## Usage

Always run from the project root.

```bash
python -m src.prepare_data                   # build data/processed/ham_28.npz (once)
python -m src.train_classifier --seed 0      # the referee CNN
python -m src.train_cvae --beta 1 --latent 16 --seed 0
python -m src.train_cgan --loss bce --seed 0
python -m src.train_ddpm --schedule linear --seed 0
python -m src.sample --model cvae --seed 0   # generate images per class
python -m src.evaluate                       # results/metrics.csv
python -m src.augment                        # results/augmentation.csv
streamlit run app/app.py                     # the demo
```

## Conventions

- **Class order:** `["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]` (alphabetical). Label = position in this list. This is **not** the label numbering of the hmnist files.
- **Images:** stored as `uint8`, shape `(N, 28, 28, 3)`. In PyTorch: `float`, shape `(N, 3, 28, 28)`, values in [-1, 1]. Generators end with `tanh`.
- **Generators:** every generator has the same method, `sample(labels)`, which returns images in [-1, 1].
- **Test set:** never used for training a generator or the classifier.

## Progress

**Week 1: foundations**
- [x] Setup: README, .gitignore, requirements, journal, folders
- [x] `src/config.py`
- [x] `src/utils.py`
- [x] `src/prepare_data.py`
- [x] `src/data.py`
- [x] `notebooks/01_exploration.ipynb`
- [x] `src/models/classifier.py` + `src/train_classifier.py`
- [x] `notebooks/02_train_kaggle.ipynb`
- [x] `app/app.py` (minimal)

**Week 2: cVAE and evaluation**
- [x] `src/models/cvae.py` + `src/train_cvae.py`
- [x] `src/sample.py`
- [x] `src/metrics.py` + `src/evaluate.py`
- [x] `app/loaders.py` + `app/views/latent.py`

**Week 3: GAN**
- [ ] `src/models/cgan.py` + `src/train_cgan.py` + `app/views/generate.py`

**Week 4: diffusion**
- [ ] `src/models/unet.py` + `src/models/ddpm.py` + `src/train_ddpm.py` + `app/views/denoise.py`

**Week 5: experiment and demo**
- [ ] `src/augment.py`
- [ ] `app/views/problem.py` + `app/views/results.py` + `notebooks/03_figures.ipynb`
- [ ] Deploy the demo and record a backup video

**Week 6: finish**
- [ ] Report, slides, final README, pinned requirements

## Disclaimer

The generated images are for research and teaching only. They are not medical images and must not be used for diagnosis.
