"""Report figures for Assignment 3, rebuilt from results/metrics.json (no training, no dataset needed).

Usage (from the repository root):  python make_report_figures.py --results results --out figs
The image montages (q2_inputs.png, q3_reconstructions.png, q4_boundaries.png) are written by run.py itself.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 3.5,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def q1_curves(m, out):
    runs = [("q1_shallow_0.001", "shallow, lr 1e-3", BLUE, "-"), ("q1_shallow_0.0002", "shallow, lr 2e-4", BLUE, "--"),
            ("q1_deep_0.001", "deep, lr 1e-3", ORANGE, "-"), ("q1_deep_0.0002", "deep, lr 2e-4", ORANGE, "--")]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.05))
    for name, lab, col, ls in runs:
        h = m["runs"][name]["history"]
        e = [r["epoch"] for r in h]
        ax[0].plot(e, [r["train_accuracy"] for r in h], ls=ls, color=col, label=lab)
        ax[1].plot(e, [r["val_accuracy"] for r in h], ls=ls, color=col, label=lab)
        ax[2].plot(e, [r["val_loss"] for r in h], ls=ls, color=col, label=lab)
        b = m["runs"][name]["best_epoch"]
        ax[1].plot(b, h[b - 1]["val_accuracy"], "o", color=col, ms=4)
    for a, t in zip(ax, ["Training accuracy", "Validation accuracy (dot = selected epoch)", "Validation loss"]):
        a.set_title(t)
        a.set_xlabel("epoch")
    ax[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "q1_curves")


def q1_memorization(m, out):
    h = m["runs"]["q1_mem"]["history"]
    e = np.arange(1, len(h) + 1)
    get = lambda k: np.array([r[k] for r in h])
    fig, a = plt.subplots(figsize=(3.4, 2.75))
    a.plot(e, get("clean_fit"), color=BLUE, label="train, clean labels (3,600)")
    a.plot(e, get("original_label_fit"), color=AQUA, label="relabelled: predicts TRUE class")
    a.plot(e, get("wrong_label_fit"), color=ORANGE, label="relabelled: predicts WRONG label")
    a.plot(e, get("val_accuracy"), color=INK, ls="--", lw=1.2, label="clean validation (5,000)")
    peak = int(get("val_accuracy").argmax()) + 1
    cross = int(np.argmax(get("wrong_label_fit") > get("original_label_fit"))) + 1
    half = int(np.argmax(get("wrong_label_fit") > 0.5)) + 1
    for x, t in [(peak, f"val peak\nep {peak}"), (cross, f"wrong > true\nep {cross}"), (half, f"wrong > 50%\nep {half}")]:
        a.axvline(x, color=MUTED, lw=0.7, ls=":")
        a.text(x + 1, 0.86, t, fontsize=6, color=MUTED, va="top")
    a.set_xlabel("epoch")
    a.set_ylabel("accuracy / agreement")
    a.set_ylim(0, 0.9)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=5.9)
    fig.tight_layout()
    save(fig, out, "q1_memorization")


def q2_bars(m, out):
    t = m["q2_test"]
    conds = ["clean", "pixel", "block"]
    names = {"clean": "clean", "pixel": "pixel shuffle", "block": "8x8 block shuffle"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.9))
    x = np.arange(3)
    w = 0.36
    for off, kind, col, lab in [(-w / 2, "mlp", BLUE, "MLP (1.55M)"), (w / 2, "cnn", ORANGE, "CNN (1.56M)")]:
        acc = [t[f"{kind}_{c}"]["accuracy"] for c in conds]
        conf_err = [t[f"{kind}_{c}"]["wrong_confidence"] for c in conds]
        b0 = ax[0].bar(x + off, acc, w * 0.94, color=col, label=lab)
        ax[0].bar_label(b0, fmt="%.3f", fontsize=6, color=INK)
        b1 = ax[1].bar(x + off, conf_err, w * 0.94, color=col, label=lab)
        ax[1].bar_label(b1, fmt="%.2f", fontsize=6, color=INK)
    for a, ttl in [(ax[0], "Test accuracy"), (ax[1], "Mean confidence on wrong predictions")]:
        a.set_xticks(x, [names[c] for c in conds])
        a.set_title(ttl)
        a.set_ylim(0, 0.92)
        a.grid(axis="x", visible=False)
    ax[0].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, out, "q2_bars")


def q3_tradeoff(m, out):
    p = m["q3_probes"]
    d = m["q3_denoising"]["test"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1, 1.35]})
    sizes = ["16", "64", "256"]
    mse = [p[s]["test_mse"] for s in sizes]
    pa = [p[s]["test_probe_accuracy"] for s in sizes]
    ax[0].plot(mse, pa, "-o", color=BLUE)
    for s, xm, ya in zip(sizes, mse, pa):
        ax[0].annotate(f"bottleneck {s}", (xm, ya), textcoords="offset points", xytext=(-58, -10) if s == "256" else (4, -10),
                       fontsize=6.5, color=INK)
    ax[0].invert_xaxis()
    ax[0].set_xlim(0.019, 0.0)
    ax[0].set_ylim(0.32, 0.44)
    ax[0].set_xticks([0.015, 0.010, 0.005, 0.0])
    ax[0].set_xlabel("test reconstruction MSE (lower is better ->)")
    ax[0].set_ylabel("frozen linear-probe test acc.")
    ax[0].set_title("Reconstruction vs class information")
    keys = [("clean", "clean\ninput"), ("corrupted", "noisy\ninput"), ("clean_ae", "clean AE\n(noisy in)"),
            ("denoiser", "denoising AE\n(noisy in)")]
    cols = [MUTED, ORANGE, BLUE, AQUA]
    b = ax[1].bar(range(4), [d[k]["accuracy"] for k, _ in keys], 0.6, color=cols)
    ax[1].bar_label(b, labels=[f"{d[k]['accuracy']:.3f}\nMSE {d[k]['mse']:.4f}" for k, _ in keys], fontsize=6, color=INK)
    ax[1].set_xticks(range(4), [lab for _, lab in keys])
    ax[1].set_ylim(0, 0.95)
    ax[1].set_title("Fixed Q2 CNN on noisy test images (sigma = 0.25)")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "q3_tradeoff")


def q4_sweep(m, out):
    sw = m["q4"]["sweep"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.05))
    for k, col, mk in [("linear", BLUE, "o"), ("rbf", ORANGE, "s"), ("poly", AQUA, "^")]:
        r = [x for x in sw if x["kernel"] == k]
        C = [x["C"] for x in r]
        lab = "poly (deg 2)" if k == "poly" else k
        ax[0].plot(C, [x["val_accuracy"] for x in r], marker=mk, color=col, label=f"{lab}, validation")
        ax[0].plot(C, [x["train_accuracy"] for x in r], marker=mk, color=col, ls=":", lw=1.0, mfc="none",
                   label=f"{lab}, train")
        ax[1].plot(C, [x["support_fraction"] for x in r], marker=mk, color=col, label=lab)
    sel = m["q4"]
    ax[0].plot(sel["selected_C"], max(x["val_accuracy"] for x in sw), "*", color=INK, ms=9, label="selected")
    for a, t in [(ax[0], "Accuracy (3,000-image SVM subset; 5,000 validation)"), (ax[1], "Support-vector fraction")]:
        a.set_xscale("log")
        a.set_xlabel("C")
        a.set_title(t)
    ax[0].set_ylim(0.25, 1.02)
    ax[0].legend(fontsize=5.6, ncol=2, loc="lower right")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, out, "q4_sweep")


def q4_pairs(m, out):
    D = m["q4"]["disagreement"]
    s = np.array([r["svm_mutual_error_rate"] for r in D])
    c = np.array([r["cluster_cross_class_fraction"] for r in D])
    fig, a = plt.subplots(figsize=(3.4, 2.45))
    a.scatter(s, c, s=12, color=BLUE, zorder=3)
    a.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    a.text(0.001, 0.505, "class-blind clustering (0.5)", fontsize=6, color=MUTED, va="bottom")
    label = {(3, 5): (-22, -12), (2, 4): (4, -2), (1, 9): (4, -3), (2, 7): (-12, 5), (4, 7): (4, -8), (0, 8): (4, -9),
             (2, 3): (4, 1)}
    for r, x, y in zip(D, s, c):
        p = tuple(r["pair"])
        if p in label:
            a.annotate(f"{CLASSES[p[0]]}/{CLASSES[p[1]]}", (x, y), textcoords="offset points", xytext=label[p],
                       fontsize=6, color=INK)
            a.scatter([x], [y], s=16, color=ORANGE, zorder=4)
    a.set_xlabel("SVM mutual test error rate (pair)")
    a.set_ylabel("same-cluster pairs crossing class")
    a.set_ylim(0, 0.58)
    a.set_xlim(-0.005, 0.19)
    fig.tight_layout()
    save(fig, out, "q4_pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = json.loads((args.results / "metrics.json").read_text())
    q1_curves(m, args.out)
    q1_memorization(m, args.out)
    q2_bars(m, args.out)
    q3_tradeoff(m, args.out)
    q4_sweep(m, args.out)
    q4_pairs(m, args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
