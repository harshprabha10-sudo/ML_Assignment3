"""Run the experiments with the official CIFAR-10 splits."""
import argparse
import itertools
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

def mlp(deep=False):
    if deep:
        return nn.Sequential(nn.Flatten(), nn.Linear(3072, 440), nn.ReLU(),
                             nn.Linear(440, 440), nn.ReLU(), nn.Linear(440, 10))
    return nn.Sequential(nn.Flatten(), nn.Linear(3072, 504), nn.ReLU(), nn.Linear(504, 10))

def cnn():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(48 * 16, 1570), nn.ReLU(), nn.Linear(1570, 216), nn.ReLU(), nn.Linear(216, 10))

class Autoencoder(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(4096, size))
        self.decoder = nn.Sequential(nn.Linear(size, 4096), nn.ReLU(),
                                     nn.Unflatten(1, (64, 8, 8)),
                                     nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                                     nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

def metrics(logits, target):
    p = logits.softmax(1)
    confidence, prediction = p.max(1)
    correct = prediction.eq(target)
    ece = 0.0
    for b in range(10):
        mask = ((confidence * 10).long().clamp(max=9) == b)
        if mask.any():
            ece += float(mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs())
    return dict(accuracy=float(correct.float().mean()),
                mean_confidence=float(confidence.mean()), ece=ece,
                wrong_confidence=float(confidence[~correct].mean()) if (~correct).any() else None,
                confusion=confusion_matrix(target, prediction, labels=range(10)).tolist())

def plot(out, name, lines, ylabel, xlabel="Epoch", logx=False):
    fig, ax = plt.subplots(figsize=(7.2, 4), layout="constrained")
    for label, (x, y) in lines.items():
        ax.plot(x, y, marker=".", label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(out / (name + ".png"), dpi=160)
    plt.close(fig)

def montage(out, name, rows, labels):
    fig, axes = plt.subplots(len(rows), len(rows[0]), squeeze=False,
                             figsize=(2 * len(rows[0]), 1.9 * len(rows)), layout="constrained")
    for row, images, titles in zip(axes, rows, labels):
        for ax, image, title in zip(row, images, titles):
            ax.imshow(image.permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.savefig(out / (name + ".png"), dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll", required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--data", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    digits = "".join(re.findall(r"[0-9]", args.roll))
    if not digits:
        parser.error("--roll must contain numeric characters")
    seed = int(digits)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    fit_seed = seed % (2**32)
    def tensors(ds):
        return torch.from_numpy(ds.data).permute(0, 3, 1, 2), torch.tensor(ds.targets)

    train_set = CIFAR10(args.data, train=True, download=not (Path(args.data) / 'cifar-10-batches-py').is_dir())
    all_x, all_y = tensors(train_set)
    train, val = train_test_split(np.arange(len(all_y)), test_size=5000, stratify=all_y,
                                  random_state=np.random.RandomState(rng.bit_generator))
    x, y, vx, vy = all_x[train], all_y[train], all_x[val], all_y[val]
    perm = torch.tensor(rng.permutation(1024))
    blocks = torch.tensor(rng.permutation(16))
    block_grid = torch.arange(1024).reshape(4, 8, 4, 8).permute(0, 2, 1, 3).reshape(16, 8, 8)
    assert np.intersect1d(train, val).size == 0
    assert np.all(np.bincount(all_y[train].numpy()) == 4500)
    assert np.all(np.bincount(all_y[val].numpy()) == 500)
    block_perm = block_grid[blocks].reshape(4, 4, 8, 8).permute(0, 2, 1, 3).reshape(-1)
    assert torch.equal(perm.sort().values, torch.arange(1024))
    assert torch.equal(block_perm.sort().values, torch.arange(1024))
    np.savez_compressed(out / "split.npz", train=train, validation=val,
                        pixel_permutation=perm.numpy(), block_permutation=block_perm.numpy())
    noise_seeds = dict(train=fit_seed+1, val=fit_seed+2, test=fit_seed+3)
    logs = {"batch_size":128, "classifier_epochs":20, "roll":args.roll, "seed":seed, "device":str(device), "torch":torch.__version__,
            "noise_sigma":.25, "noise_seeds":noise_seeds, "runs":{}}

    log_path = out / "metrics.json"
    if log_path.exists():
        previous = json.loads(log_path.read_text())
        if any(previous[k] != logs[k] for k in ("roll", "seed", "device", "torch")):
            raise ValueError("Resume requires the same roll, device and PyTorch version.")
        logs["runs"] = previous["runs"]

    def augment(a, cond):
        a = a.float() / 255 if a.dtype == torch.uint8 else a
        lookup = {"pixel":perm, "block":block_perm}
        if cond in lookup:
            return a.flatten(2)[:, :, lookup[cond]].reshape_as(a)
        return a

    def predict(model, a, cond="clean", encoder=False):
        model.eval()
        fn = model.encoder if encoder else model
        with torch.inference_mode():
            return torch.cat([fn(augment(b, cond).to(device)).cpu() for b in a.split(128)])

    def fit(name, factory, a=x, b=y, va=vx, vb=vy, cond="clean", lr=.001,
            epochs=20, reconstruction=False, noisy_indices=None, original_labels=None):
        torch.manual_seed(fit_seed)
        model = factory().to(device)
        if name in logs["runs"]:
            model.load_state_dict(torch.load(out / (name + ".pt"), map_location=device))
            return model.cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss() if reconstruction else nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(a, b), batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(fit_seed))
        history, best = [], -math.inf
        for epoch in range(1, epochs + 1):
            model.train()
            for image, target in loader:
                target = augment(target, "clean") if reconstruction else target
                logits = model(augment(image, cond).to(device))
                loss = loss_fn(logits, target.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            valid = predict(model, va, cond)
            if reconstruction:
                row = dict(epoch=epoch, val_mse=float((valid - augment(vb, "clean")).square().mean()))
                score = -row["val_mse"]
            else:
                train_logits = predict(model, a, cond)
                pred = train_logits.argmax(1)
                row = dict(epoch=epoch, train_accuracy=float(pred.eq(b).float().mean()),
                           val_accuracy=float(valid.argmax(1).eq(vb).float().mean()),
                           train_loss=float(loss_fn(train_logits, b)), val_loss=float(loss_fn(valid, vb)))
                if noisy_indices is not None:
                    clean_mask = torch.ones(len(b), dtype=torch.bool)
                    clean_mask[noisy_indices] = False
                    row.update(clean_fit=float(pred[clean_mask].eq(b[clean_mask]).float().mean()),
                               wrong_label_fit=float(pred[noisy_indices].eq(b[noisy_indices]).float().mean()),
                               original_label_fit=float(pred[noisy_indices].eq(original_labels[noisy_indices]).float().mean()))
                score = row["val_accuracy"]
            if score > best:
                best = score
                best_epoch = epoch
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append(row)
            print(name, row, flush=True)
        model.load_state_dict(state)
        torch.save(state, out / (name + ".pt"))
        logs["runs"][name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                 epochs=epochs, lr=lr, best_epoch=best_epoch, history=history)
        (out / "metrics.json").write_text(json.dumps(logs, indent=2))
        return model.cpu()

    q1 = {}
    for deep, lr in itertools.product((False, True), (.001, .0002)):
        name = f"q1_{'deep' if deep else 'shallow'}_{lr}"
        q1[name] = fit(name, lambda: mlp(deep), lr=lr)
    q1_names = list(q1)
    for key in ("accuracy", "loss"):
        for split in ("train", "val"):
            field = f"{split}_{key}"
            plot(out, "q1_" + field, {n: (range(1,21), [r[field] for r in logs["runs"][n]["history"]]) for n in q1}, field)

    sub, _ = train_test_split(np.arange(len(y)), train_size=4500, stratify=y, random_state=fit_seed)
    ny = y[sub].clone()
    flips = rng.choice(len(sub), 900, replace=False)
    ny[flips] = (ny[flips] + torch.tensor(rng.integers(1, 10, len(flips)))) % 10
    fit("q1_mem", mlp, x[sub], ny, epochs=100, noisy_indices=flips, original_labels=y[sub])
    history = logs["runs"]["q1_mem"]["history"]
    plot(out, "q1_memorization", {k:(range(1,101), [r[k] for r in history]) for k in
          ("clean_fit", "wrong_label_fit", "original_label_fit", "val_accuracy")}, "Accuracy")

    q2 = {("mlp","clean"):q1[q1_names[0]], ("cnn","clean"):fit("q2_cnn_clean", cnn)}
    for kind, cond in itertools.product(("mlp", "cnn"), ("pixel", "block")):
        q2[kind, cond] = fit(f"q2_{kind}_{cond}", mlp if kind == "mlp" else cnn, cond=cond)

    def noisy(a, split):
        g = torch.Generator().manual_seed(noise_seeds[split])
        return (a.float() / 255 + .25 * torch.randn(a.shape, generator=g)).clamp_(0, 1)

    aes = {size:fit(f"q3_ae{size}", lambda:Autoencoder(size), x, x, vx, vx,
                    reconstruction=True) for size in (16, 64, 256)}
    selected = min(aes, key=lambda s:min(r["val_mse"] for r in logs["runs"][f"q3_ae{s}"]["history"]))
    denoiser = fit("q3_denoiser", lambda:Autoencoder(selected), noisy(x, "train"), x,
                   noisy(vx, "val"), vx, reconstruction=True)
    probes = {}
    for size, model in aes.items():
        z = predict(model.to(device), x, encoder=True).numpy()
        scale = StandardScaler().fit(z)
        clf = LogisticRegression(C=1, max_iter=1000, random_state=fit_seed).fit(scale.transform(z), y)
        probes[size] = scale, clf
        model.cpu()

    # Q4 uses the clean validation-selected CNN's penultimate activations.
    feature_net = q2["cnn","clean"][:-1].to(device)
    f, fv = (predict(feature_net, a).numpy() for a in (x, vx))
    scaler = StandardScaler().fit(f)
    f, fv = scaler.transform(f), scaler.transform(fv)
    svm_idx, _ = train_test_split(np.arange(len(y)), train_size=3000, stratify=y, random_state=fit_seed)
    np.savez_compressed(out / "split.npz", train=train, validation=val, pixel_permutation=perm.numpy(),
                        block_permutation=block_perm.numpy(), memorization_train_positions=sub,
                        memorization_flip_positions=flips, memorization_labels=ny.numpy(), svm_train_positions=svm_idx)
    sf, sy = f[svm_idx], y[svm_idx].numpy()
    sweep, svm, best = [], None, -1
    c_values = [.01, .1, 1, 10, 100]
    for kernel, c in itertools.product(("linear", "rbf", "poly"), c_values):
        candidate = SVC(C=c, kernel=kernel, degree=2, decision_function_shape="ovo").fit(sf, sy)
        row = dict(kernel=kernel, C=c, train_accuracy=candidate.score(sf, sy),
                   val_accuracy=candidate.score(fv, vy), support_fraction=len(candidate.support_)/len(sf))
        sweep.append(row)
        if row["val_accuracy"] > best:
            svm, best = candidate, row["val_accuracy"]
    for metric in ("val_accuracy", "support_fraction"):
        plot(out, "q4_" + metric, {k:(c_values, [r[metric] for r in sweep if r["kernel"] == k])
             for k in ("linear","rbf","poly")}, metric, "C", logx=True)

    cluster_scores, cluster, best = {}, None, -math.inf
    for k in (8, 10, 15, 20):
        candidate = KMeans(n_clusters=k, n_init=5, random_state=fit_seed).fit(f)
        assignment = candidate.predict(fv)
        score = float(silhouette_score(fv, assignment))
        cluster_scores[k] = score
        if score > best:
            cluster, best = candidate, score

    # Test data enter after all model and hyperparameter choices are fixed.
    tx, ty = tensors(CIFAR10(args.data, train=False, download=False))
    logs["q1_test"] = {n:metrics(predict(model.to(device), tx), ty) for n, model in q1.items()}
    logs["q2_test"] = {f"{kind}_{cond}":metrics(predict(model.to(device), tx, cond), ty)
                       for (kind,cond), model in q2.items()}
    example = vx[:5]
    montage(out, "q2_inputs", [augment(example,c) for c in ("clean","pixel","block")],
            [[f"{c} • {train_set.classes[int(v)]}" for v in vy[:5]] for c in ("clean","pixel","block")])
    corrupted = noisy(tx, "test")
    clean_ae = aes[selected].to(device)
    restored_clean = predict(clean_ae, corrupted)
    restored_denoised = predict(denoiser.to(device), corrupted)
    target = augment(tx, "clean")
    fixed = q2["cnn","clean"]
    variants = dict(clean=tx, corrupted=corrupted, clean_ae=restored_clean, denoiser=restored_denoised)
    logs["q3_denoising"] = {"selected_size":selected,
        "test":{k:dict(mse=float((augment(a,"clean")-target).square().mean()),
                       **metrics(predict(fixed, a), ty)) for k,a in variants.items()}}
    montage(out, "q3_reconstructions", [augment(a[:5],"clean") for a in variants.values()],
            [[k]*5 for k in variants])
    logs["q3_probes"] = {}
    for size, model in aes.items():
        model.to(device)
        scale, clf = probes[size]
        test_code = predict(model, tx, encoder=True)
        logs["q3_probes"][size] = dict(
            test_mse=float((predict(model, tx)-target).square().mean()),
            test_probe_accuracy=clf.score(scale.transform(test_code.numpy()), ty))
        model.cpu()

    ft = scaler.transform(predict(feature_net, tx).numpy())
    svm_pred = svm.predict(ft)
    cm = confusion_matrix(ty, svm_pred, labels=range(10))
    test_scores = svm.decision_function(ft)
    train_scores = svm.decision_function(sf)
    pairs = list(itertools.combinations(range(10),2))
    hardest = sorted(pairs, key=lambda ab:-(cm[ab]+cm[ab[::-1]]))[:3]
    pair_details = []
    rows, titles = [], []
    offsets = np.r_[0, svm.n_support_.cumsum()]
    for a, b in hardest:
        column = pairs.index((a,b))
        sv_positions = np.r_[np.flatnonzero(svm.dual_coef_[b-1,offsets[a]:offsets[a+1]])+offsets[a],
                             np.flatnonzero(svm.dual_coef_[a,offsets[b]:offsets[b+1]])+offsets[b]]
        support = svm.support_[sv_positions]
        support = support[np.argsort(abs(train_scores[support,column]))[:3]]
        eligible = np.flatnonzero(np.isin(ty.numpy(), [a,b]))
        near = eligible[np.argsort(abs(test_scores[eligible,column]))[:3]]
        pair_details.append(dict(pair=[a,b], mutual_errors=int(cm[a,b]+cm[b,a]),
                                 train_indices=train[svm_idx[support]].tolist(),
                                 train_scores=train_scores[support,column].tolist(),
                                 test_indices=near.tolist(), test_scores=test_scores[near,column].tolist(),
                                 test_labels=ty[near].tolist(), test_predictions=svm_pred[near].tolist()))
        rows.append(list(augment(x[svm_idx[support]],"clean")) + list(augment(tx[near],"clean")))
        titles.append([f"SV {train_set.classes[sy[i]]}\n{train_scores[i,column]:+.3f}" for i in support] +
                      [f"Test {train_set.classes[ty[i]]} → {train_set.classes[svm_pred[i]]}\n{test_scores[i,column]:+.3f}" for i in near])
    blank = torch.ones(3, 32, 32)
    for row, title in zip(rows, titles):
        for _ in range(6 - len(row)):
            row.insert(0, blank)
            title.insert(0, "")
    montage(out, "q4_boundaries", rows, titles)
    groups = cluster.predict(ft)
    contingency = contingency_matrix(ty, groups)
    disagreements = []
    for a,b in pairs:
        cross = 2 * np.dot(contingency[a], contingency[b])
        n = contingency[a] + contingency[b]
        same_group_pairs = np.dot(n,n) - n.sum()
        disagreements.append(dict(pair=[a,b], svm_mutual_error_rate=float((cm[a,b]+cm[b,a])/(cm[a].sum()+cm[b].sum())),
                                  cluster_cross_class_fraction=float(cross/max(1,same_group_pairs))))
    logs["q4"] = dict(sweep=sweep, selected_kernel=svm.kernel, selected_C=svm.C,
                      test_accuracy=float(np.mean(svm_pred==ty.numpy())), test_confusion=cm.tolist(),
                      difficult_pairs=pair_details, silhouette=cluster_scores, selected_k=cluster.n_clusters,
                      test_ari=adjusted_rand_score(ty,groups), test_nmi=normalized_mutual_info_score(ty,groups),
                      contingency_class_rows=contingency.tolist(), disagreement=disagreements)
    (out / "metrics.json").write_text(json.dumps(logs, indent=2))
    np.savez_compressed(out / "predictions.npz", test_labels=ty.numpy(), svm=svm_pred, clusters=groups,
                        ovo_decision_scores=test_scores)

if __name__ == "__main__":
    main()
