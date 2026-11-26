# Machine Learning Assignment 3: MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR-10

**Name:** Harsh Prabha Bhaina  **Roll number:** M26ECM503  **Institute:** IIT Jodhpur
**Seed:** `26503` (all numeric characters of the roll number, in order)

The repository link and final commit hash are on page 1 of the report (a commit cannot contain its own hash).

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | All experiments (Q1–Q4): split, MLPs, CNNs, autoencoders, probes, SVM sweep, k-means |
| `results/split.npz` | Saved split: train/validation indices, pixel and block permutations, memorisation subset and flipped labels, SVM subset |
| `results/metrics.json` | Every logged number: per-epoch histories, selected epochs, parameter counts, test metrics, SVM sweep, clustering |
| `results/predictions.npz` | Test labels, SVM predictions, cluster assignments, one-vs-one decision scores |
| `results/*.png` | Figures written by `run.py` (curves and image montages) |
| `training.log` | Final console log of the full run |
| `make_report_figures.py` | Figure script for the report plots (reads `results/metrics.json` only) |
| `render.py` | Optional HTML walkthrough of the results |
| `report/` | CVPR report source (`main.tex`, bibliography, style files, figures) |
| `requirements.txt`, `environment-lock.txt` | Environment (exact versions in the lock file) |

## Configuration

All hyperparameters are fixed in `run.py` and recorded in the header of `results/metrics.json`:
batch 128, Adam, 20 epochs per classifier and autoencoder (100 for the memorisation run), learning rates 1e-3 and 2e-4 (Q1),
noise sigma 0.25 with noise seeds 26504/26505/26506, SVM C in {0.01, 0.1, 1, 10, 100} for linear/RBF/poly(degree 2)
on a fixed stratified 3,000-image subset, and k-means k in {8, 10, 15, 20} chosen by validation silhouette.

## Reproduce the experiments

Python 3.12. The recorded run used an NVIDIA T1000 with the CUDA 12.6 PyTorch wheels. CPU works but is much slower
and may give slightly different floating-point results.

```sh
python -m venv .venv
# activate the environment for your shell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r environment-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -u run.py --device cuda --roll M26ECM503 --data data --out results > training.log 2>&1
python make_report_figures.py --results results --out report/figs
```

`run.py` downloads CIFAR-10 unless `cifar-10-batches-py/` already exists under `--data`. Rerunning with the same
`--out` restores finished stages from their checkpoints (`results/*.pt`, kept out of Git because of their size).

## Build the report

```sh
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Before the final build, set `\repourl` and `\commithash` near the top of `report/main.tex`.

## Headline test results

| Model | Test accuracy |
|---|---|
| Shallow / deep MLP (validation-selected lr) | 50.02% / 52.59% |
| CNN, clean / pixel-shuffled / 8x8 block-shuffled | 71.57% / 48.52% / 62.10% |
| Fixed CNN on noisy / clean-AE / denoising-AE inputs | 24.70% / 52.65% / 51.38% |
| Linear SVM (C = 0.01) on CNN features | 71.73% |
| k-means (k = 8) on CNN features | ARI 0.346, NMI 0.493 |
