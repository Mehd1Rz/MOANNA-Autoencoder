# Moanna Re-implementation

Re-implementation of the breast-cancer molecular subtyping models from:

> Lupat et al., **"Moanna: Multi-Omics Autoencoder-Based Neural Network Algorithm for Predicting Breast Cancer Subtypes"**, *IEEE Access*, 2023.
> Paper: https://ieeexplore.ieee.org/document/10029336

The project contains a Keras semi-supervised multi-task autoencoder (Moanna-style) and two classical baselines (PCA + KNN and an RBF-kernel SVM), all evaluated on the same held-out test set.

---

## Prediction tasks

Every model is trained on the same six label sets:

| Label | Type | Classes |
|---|---|---|
| `ERStatus` | Binary | 2 |
| `HER2Status` | Binary | 2 |
| `Pam50Subtype` | Multi-class | 4 |
| `Pam50SubtypeSingleLuminal` | Multi-class | 3 |
| `BasalNonBasal` | Binary | 2 |
| `Luminal` | Multi-class | 3 |

## Repository structure

```
MOANNA-Autoencoder/
├── data/
│   ├── moanna_training_data.gz        <- you download this (see "Data" below)
│   └── extracted/                     <- created automatically by the first cell
│       └── moanna_data/
│           ├── training/              <- training + validation .tsv files
│           └── testing/               <- test .tsv files
├── 01_autoencoder_multitask.ipynb     <- Keras multi-task autoencoder
├── 02_baseline_pca_knn.ipynb          <- PCA (100 components) + KNN, k = 1..14
├── 03_baseline_svm.ipynb              <- SVM (RBF kernel), one model per label
└── README.md
```

## Data

The dataset is published on Zenodo: **https://doi.org/10.5281/zenodo.4326602**

1. Download the file **`moanna_training_data.gz`** from the Zenodo record.
2. Save it in the `data/` folder of this repo (recommended).
   - If you don't, the first cell of each notebook searches the current folder and its parent folders for a folder named `data`, so keeping the file anywhere inside a `data/` folder above the notebook also works.
3. Run the first cell of any notebook. It will:
   - locate `moanna_training_data*` inside `data/`,
   - extract the archive to `data/extracted/moanna_data/`,
   - find the `.tsv` files automatically (using a recursive search, so the exact sub-folder layout does not matter).

After extraction the data is organised as:

| Folder | Contents |
|---|---|
| `data/extracted/moanna_data/training/` | training data + labels **and** validation data + labels |
| `data/extracted/moanna_data/testing/` | test data + labels |

Files used by the notebooks:

```
moanna_training_data.tsv    moanna_training_label.tsv
moanna_validation_data.tsv  moanna_validation_label.tsv
moanna_testing_data.tsv     moanna_testing_label.tsv
```

Dataset dimensions (as printed by the notebooks): **1,182** training samples, **631** test samples, **46,776** features per sample.

> The extraction only runs if `data/extracted/` does not exist yet. To re-extract, delete that folder.

## Installation

```bash
git clone https://github.com/Mehd1Rz/MOANNA-Autoencoder.git
cd MOANNA-Autoencoder
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib tensorflow jupyter
```

Then launch `jupyter notebook` and open the notebooks in order.

## Notebooks

### 01 - Multi-task autoencoder (`01_autoencoder_multitask.ipynb`)

A Keras subclassed model combining a denoising-style autoencoder with six classification heads trained jointly (semi-supervised / multi-task learning).

- **Encoder:** Dense(256) → BatchNorm → tanh → Dense(128) → BatchNorm → tanh → Dense(64, tanh) → Dropout(0.5)
- **Classifier trunk:** Dense(40, ReLU) on the 64-dim code, feeding six heads:
  - sigmoid (1 unit) for `ERStatus`, `HER2Status`, `BasalNonBasal`
  - softmax for `Pam50Subtype` (4), `Pam50SubtypeSingleLuminal` (3), `Luminal` (3)
- **Decoder:** Dense(128) → BatchNorm → tanh → Dense(256) → BatchNorm → tanh → Dropout(0.5) → Dense(input_dim, tanh)
- **Losses:** binary cross-entropy (binary heads), categorical cross-entropy (multi-class heads), MSE (reconstruction)
- **Optimizer:** SGD, learning rate 0.005, momentum 0.9
- **Training:** 100 epochs, batch size 32, 10% validation split

The model has 7 outputs: the six labels plus the reconstructed input.

### 02 - PCA + KNN baseline (`02_baseline_pca_knn.ipynb`)

- Standardises the combined training + validation data (`StandardScaler`), reduces it to **100 principal components** (PCA), and applies the same transform to the test set.
- Trains `KNeighborsClassifier` for **k = 1 to 14** on each of the six labels and records **test accuracy**.
- Prints the mean accuracy over the six labels for every k.

### 03 - SVM baseline (`03_baseline_svm.ipynb`)

- One `sklearn.svm.SVC` (default RBF kernel) per label, trained on the training split on the raw 46,776 features.
- Reports **weighted F1** and **accuracy** on the test set, highlights the best and worst label, and plots a bar chart.

## Results

### SVM (RBF kernel) - test set (631 samples)

| Label | F1 (weighted) % | Accuracy % |
|---|---:|---:|
| ERStatus | 94.79 | 94.77 |
| HER2Status | 74.55 | 80.67 |
| Pam50Subtype | 81.46 | 80.82 |
| Pam50SubtypeSingleLuminal | 94.27 | 94.77 |
| BasalNonBasal | **99.05** | **99.05** |
| Luminal | 82.19 | 81.62 |
| **Average** | **87.72** | **88.62** |

Key findings:

- **Best label:** `BasalNonBasal` (99.05% F1) - basal vs. non-basal separates very cleanly.
- **Worst label:** `HER2Status` (74.55% F1). Its accuracy (80.67%) is noticeably higher than its F1, which points to class imbalance (few HER2-positive samples).
- `ERStatus` and `Pam50SubtypeSingleLuminal` are also strong (about 94-95%).
- The 4-class `Pam50Subtype` and the 3-class `Luminal` tasks sit around 81-82%.

### PCA + KNN - test accuracy (%)

PCA (100 components) + `KNeighborsClassifier`, k = 1..14, evaluated on the 631-sample test set.

| k | ERStatus | HER2Status | Pam50Subtype | Pam50SubtypeSingleLuminal | BasalNonBasal | Luminal | Average |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 90.65 | 78.92 | 69.89 | 87.16 | 96.35 | 72.27 | 82.54 |
| 2 | 90.17 | 77.81 | 72.42 | 86.69 | 95.72 | 75.12 | 82.99 |
| 3 | 91.76 | 80.51 | 72.11 | 90.65 | 97.31 | 74.48 | 84.47 |
| 4 | 90.81 | **80.67** | 75.12 | 91.92 | 97.62 | 77.02 | 85.53 |
| 5 | **91.92** | 80.19 | 75.12 | 91.60 | 98.26 | 76.55 | 85.60 |
| 6 | 90.33 | **80.67** | 75.75 | 91.44 | **98.57** | 77.97 | 85.79 |
| 7 | 91.60 | 79.40 | 77.02 | 91.60 | **98.57** | 78.45 | 86.11 |
| 8 | 91.28 | 79.71 | 77.34 | **92.08** | 98.42 | 78.45 | 86.21 |
| 9 | 91.28 | 79.08 | 77.97 | 91.44 | 97.94 | 78.92 | 86.11 |
| 10 | 90.81 | 79.71 | 77.50 | 91.92 | 98.10 | 79.40 | 86.24 |
| 11 | 91.13 | 78.92 | **78.29** | 91.28 | 97.62 | **79.56** | 86.13 |
| 12 | 90.81 | 79.71 | 77.81 | 91.60 | 98.10 | **79.56** | **86.27** |
| 13 | 90.97 | 79.40 | 76.86 | 90.81 | 97.46 | 78.29 | 85.63 |
| 14 | 90.33 | 79.56 | 76.70 | 91.28 | 97.78 | 78.13 | 85.63 |

**Best k by average accuracy: k = 12 (86.27%)**, with k = 10 (86.24%) and k = 8 (86.21%) practically identical.

Best k per label (ties listed together):

| Label | Best k | Accuracy % |
|---|--:|--:|
| ERStatus | 5 | 91.92 |
| HER2Status | 4 and 6 | 80.67 |
| Pam50Subtype | 11 | 78.29 |
| Pam50SubtypeSingleLuminal | 8 | 92.08 |
| BasalNonBasal | 6 and 7 | 98.57 |
| Luminal | 11 and 12 | 79.56 |

Key findings:

- Accuracy is lowest at k = 1-2 (about 82.5-83%) and rises to a plateau of roughly 85.5-86.3% for k >= 4. Larger k mainly helps the multi-class labels (`Pam50Subtype`, `Luminal`).
- `ERStatus`, `HER2Status` and `BasalNonBasal` barely change with k once k >= 3.
- As with SVM, `BasalNonBasal` is the easiest label (up to 98.57%) and `HER2Status` the hardest (about 80%).
- Differences of a few tenths of a percent between neighbouring k values (for example k = 8, 10, 12) are within noise on a 631-sample test set (one sample is about 0.16%).

### Multi-task autoencoder - training log (100 epochs)

The notebook trains with `validation_split=0.1` (about 119 held-out samples) and does **not** evaluate on the test set, so the numbers below are **validation** results at epoch 100 and are not directly comparable with the test-set tables above.

| Label | Train acc % | Validation acc % |
|---|--:|--:|
| ERStatus | 100.00 | 99.16 |
| HER2Status | 100.00 | 97.48 |
| Pam50Subtype | 100.00 | 86.55 |
| Pam50SubtypeSingleLuminal | 100.00 | 91.60 |
| BasalNonBasal | 100.00 | 95.80 |
| Luminal | 100.00 | 89.08 |

Loss at epoch 100: total train loss 0.4533 vs. validation loss 2.7589; reconstruction (MSE) 0.4509 train / 0.4537 validation.

Key findings:

- The classification heads fit the training data perfectly by epoch ~12 (100% accuracy on every head).
- Validation accuracy plateaus early (about epoch 5-8) and then stays essentially flat while validation loss climbs from its minimum of about 1.68 (epoch 2) to 2.76 (epoch 100). The model is **overfitting**: confident but not better on unseen samples. Early stopping or stronger regularisation would likely help.
- The reconstruction loss barely moves (about 0.462 to 0.451), so the decoder learns little beyond the mean; the classification heads dominate training.
- As in the baselines, the 4-class `Pam50Subtype` and 3-class `Luminal` heads are the weakest (about 87-89% validation).
- Validation accuracy on `ERStatus` and `HER2Status` looks higher than the baselines' test accuracy, while `BasalNonBasal` (95.80%) is actually below them (about 98-99%). The validation set is small (~119 samples) and is simply the last 10% of the training file, so treat all of this as indicative only. Adding `autoencoder.evaluate(...)` on the test set would give a fair comparison.

### Model comparison (test set, accuracy %)

KNN uses the best k per label.

| Label | KNN (best k) | SVM |
|---|--:|--:|
| ERStatus | 91.92 | **94.77** |
| HER2Status | 80.67 | 80.67 |
| Pam50Subtype | 78.29 | **80.82** |
| Pam50SubtypeSingleLuminal | 92.08 | **94.77** |
| BasalNonBasal | 98.57 | **99.05** |
| Luminal | 79.56 | **81.62** |
| Average | 86.27 (k = 12, single k for all labels) | **88.62** |

SVM beats PCA + KNN on average (88.62% vs. 86.27%) and on five of six labels; `HER2Status` is an exact tie (80.67%). Note that the best k for KNN was picked by looking at test accuracy, which makes the KNN numbers slightly optimistic.

> **Note:** the models are not scored identically. SVM is trained on the training split only and reports weighted F1 + accuracy; KNN is trained on training + validation and reports accuracy only; the autoencoder reports validation accuracy from its training log.

## Reference

Lupat, R. et al. *Moanna: Multi-Omics Autoencoder-Based Neural Network Algorithm for Predicting Breast Cancer Subtypes.* IEEE Access, 2023. https://ieeexplore.ieee.org/document/10029336

Data: https://doi.org/10.5281/zenodo.4326602
