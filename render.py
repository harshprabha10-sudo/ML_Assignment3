"""Render the standalone walkthrough and CVPR tables from the same results."""
import argparse
import base64
import html
import json
import os
import re
from pathlib import Path

CLASSES = "airplane automobile bird cat deer dog frog horse ship truck".split()

def fmt(x):
    if x is None:
        return "—"
    return f"{x:.5f}" if isinstance(x, float) else str(x)

def escape_tex(x):
    lookup = {"_":r"\_", "&":r"\&", "%":r"\%", "#":r"\#", "—":"--"}
    return "".join(lookup.get(c,c) for c in str(x))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("walkthrough.html"))
    p.add_argument("--name", default="HARSH PRABHA BHAINA")
    p.add_argument("--roll", default="M26ECM503")
    p.add_argument("--repo", default="Pending")
    p.add_argument("--commit", default="Pending")
    args = p.parse_args()
    source = args.results / "metrics.json"
    data = json.loads(source.read_text()) if source.exists() else {}
    roll = data.get("roll", args.roll)
    runs = data.get("runs", {})
    observations_path = args.results / "observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8")) if observations_path.exists() else {}
    evidence = []
    tex = []

    def table(headers, rows):
        rows = [[fmt(a) for a in row] for row in rows]
        evidence.append("<div class='scroll'><table><thead><tr>" +
                        "".join(f"<th>{html.escape(c)}</th>" for c in headers) +
                        "</tr></thead><tbody>" + "".join("<tr>" +
                        "".join(f"<td>{html.escape(c)}</td>" for c in row) +
                        "</tr>" for row in rows) + "</tbody></table></div>")
        row_end = r" \\" + "\n"
        table_body = row_end.join(" & ".join(map(escape_tex, row)) for row in rows)
        wide = len(headers) > 12
        if wide:
            tex.append(r"\begin{table*}[t]\centering" + "\n")
        width = r"\textwidth" if wide else r"\columnwidth"
        tex.append(r"\par\noindent\resizebox{" + width + "}{!}{"
                   + r"\begin{tabular}{" + "l" + "r"*(len(headers)-1) + "}\n"
                   + "\\toprule\n" + " & ".join(map(escape_tex, headers)) + row_end
                   + "\\midrule\n" + table_body + row_end
                   + "\\bottomrule\n\\end{tabular}}\\par\n")

        if wide:
            tex.append(r"\end{table*}" + "\n")

    def figure(key, caption):
        path = args.results / (key + ".png")
        if not path.exists():
            return
        encoded = base64.b64encode(path.read_bytes()).decode()
        caption_html = html.escape(caption)
        evidence.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{caption_html}"><figcaption>{caption_html}</figcaption></figure>')
        relative = Path(os.path.relpath(path, args.out.parent)).as_posix()
        wide = key in {"q4_boundaries", "q3_reconstructions", "q2_inputs"}
        env = "figure*" if wide else "figure"
        width = r"\textwidth" if wide else r"\columnwidth"
        tex.append(f"\\begin{{{env}}}[t]\\centering\n"
                   f"\\includegraphics[width={width}]{{{relative}}}\n"
                   f"\\caption{{{escape_tex(caption)}}}\\end{{{env}}}\n")

    def section(n, title, copy):
        sections.append(f'<section id="q{n}"><small>QUESTION {n}</small><h2>{title}</h2>{copy}'
                        + ("".join(evidence) if evidence else "<p class='pending'>Measured evidence will appear here after training and evaluation.</p>")
                        + "</section>")
        if data:
            findings = re.findall(r"<p>(.*?)</p>", copy, flags=re.S)
            if findings and evidence:
                # The measured summary is appended after the experimental explanation.
                count = {1:2, 2:2, 3:2, 4:3+len(observations.get("q4", []))}[n]
                tex[:0] = [escape_tex(html.unescape(paragraph.strip())) + r"\par" for paragraph in findings[-count:]]
            (args.out.parent / f"q{n}-evidence.tex").write_text("\n".join(tex) or "Results pending.", encoding="utf-8")
        evidence.clear()
        tex.clear()

    sections = []
    if "q1_test" in data:
        table(["Model", "Parameters", "Epoch", "Test accuracy"], [
            (n, runs[n]["parameters"], runs[n]["best_epoch"], v["accuracy"]) for n,v in data["q1_test"].items()])
        a,b,c,d = [v["accuracy"] for v in data["q1_test"].values()]
        copy = f"<p>Deep minus shallow test accuracy was {100*(c-a):+.2f} points at 0.001 and {100*(d-b):+.2f} points at 0.0002.</p>"
        depth_consistent = (c > a and d > b) or (c < a and d < b)
        depth_comment = ("The direction of the depth effect held at both learning rates."
                         if depth_consistent else
                         "The direction of the depth effect depended on the learning rate or tied at one rate.")
        copy = copy[:-4] + " " + depth_comment + "</p>"
        h = runs["q1_mem"]["history"]
        peak = max(h, key=lambda row:row["val_accuracy"])
        onset = next((r for r in h if r["wrong_label_fit"] >= .50), None)
        final = h[-1]
        timing = (
            f"Replacement-label accuracy first crossed 50% at epoch {onset['epoch']}. "
            if onset else "Replacement-label accuracy never crossed 50% in this run. "
        )
        copy += (f"<p>Clean validation accuracy peaked at {peak['val_accuracy']:.2%} at epoch {peak['epoch']}. "
                 f"{timing}At epoch {peak['epoch']}, replacement-label fit was {peak['wrong_label_fit']:.2%} "
                 f"and original-label fit on those examples was {peak['original_label_fit']:.2%}. By the last epoch, "
                 f"replacement-label fit was {final['wrong_label_fit']:.2%} and clean validation accuracy was "
                 f"{final['val_accuracy']:.2%}.</p>")
        if onset and onset["epoch"] > peak["epoch"]:
            copy = copy[:-4] + " On this run, the validation peak preceded majority fit to the wrong labels.</p>"
        for k in ("train_accuracy","val_accuracy","train_loss","val_loss","memorization"):
            figure("q1_"+k, k.replace("_"," ").capitalize())
    else:
        copy = ""
    section(1,"Does depth help, and when does the MLP memorize?", """
      <p>One reason a shallow MLP may struggle is that it has only one stage of nonlinear processing to combine pixel cues into features. We compare a 504-unit hidden layer with two 440-unit hidden layers. Each model gets the same data, Adam optimizer, batch size of 128, and 20 epochs. Both run at 0.001 and 0.0002. Comparing depths at each learning rate helps distinguish an architecture effect from sensitivity to step size.</p>
      <p>In the memorization run, 900 of 4,500 labels are deliberately changed. Think of a horse assigned the label truck. A model that learns useful structure may initially call it a horse, then learn the assigned label as training continues. Read the clean validation, original-label, and wrong-label curves together. If the curves rise together, there is weaker evidence for a distinct early learning phase. If wrong-label fit stays low, the run provides limited evidence about memorization.</p>
    """+copy)
    if "q2_test" in data:
        q2 = data["q2_test"]
        table(["Classifier / condition","Parameters","Test accuracy","Wrong confidence","ECE"], [
            (k,runs["q1_shallow_0.001" if k=="mlp_clean" else "q2_"+k]["parameters"],
             v["accuracy"], v["wrong_confidence"],v["ece"]) for k,v in q2.items()])
        table(["Model / condition","True class","Predicted class","Errors"], [
            (k, CLASSES[a], CLASSES[b], count) for k,v in q2.items()
            for count,a,b in [max((n,a,b) for a,row in enumerate(v["confusion"]) for b,n in enumerate(row) if a!=b)]])
        gaps = [100*(q2["cnn_"+c]["accuracy"]-q2["mlp_"+c]["accuracy"]) for c in ("clean","pixel","block")]
        copy = f"<p>The CNN minus MLP accuracy gap was {gaps[0]:+.2f} points on clean images, {gaps[1]:+.2f} after pixel shuffling, and {gaps[2]:+.2f} after block shuffling.</p>"
        result = ("The relative gap narrowed with pixel shuffling and widened again with intact patches. "
                  "That pattern supports the role of local neighborhoods."
                  if gaps[1] < gaps[0] and gaps[2] > gaps[1] else
                  "The gaps did not follow both predictions. This comparison gives mixed evidence for the locality explanation.")
        copy += (f"<p>{result} Block shuffling preserves patch interiors while changing patch boundaries and positions. "
                 f"On the CNN's wrong predictions, average confidence was {q2['cnn_clean']['wrong_confidence']:.2%} "
                 f"on clean inputs and {q2['cnn_pixel']['wrong_confidence']:.2%} after pixel shuffling.</p>")
        figure("q2_inputs","Validation examples under the three input conditions.")
    else:
        copy = ""
    section(2,"What do local neighborhoods contribute?", """
      <p>The clean MLP is reused from Q1. Both classifiers train under the same update budget with similar parameter counts. The CNN can reuse edge and texture detectors across locations. Its computation per batch differs from the MLP's.</p>
      <figure aria-label="A toy example showing pixel and patch shuffles">
      <div class="tiles">
      <div><strong>Original</strong><pre>A A B B
A A B B
C C D D
C C D D</pre></div>
      <div><strong>Pixels shuffled</strong><pre>C B A D
A D C B
D A B C
B C D A</pre></div>
      <div><strong>Patches shuffled</strong><pre>D D A A
D D A A
B B C C
B B C C</pre></div>
      </div><figcaption>Illustration using a 4 × 4 grid. Each letter represents a pixel color. The same four colors appear four times in all three grids.</figcaption></figure>
      <p>Pixel shuffling moves each RGB triplet as a unit. A single permutation is shared across every image. A dense layer can absorb that permutation into its input weights, whereas a convolution still mixes whatever now lies nearby. Block shuffling moves whole 8 × 8 patches. If local cues matter more than whole-image layout, the CNN's relative performance should recover when patches stay intact. The wrong-prediction confidence and top confusion in each condition help show where the model still makes confident mistakes.</p>
    """+copy)
    if "q3_probes" in data:
        q3 = data["q3_denoising"]
        table(["Latent size","Parameters","Val MSE","Test MSE","Probe accuracy"], [
            (int(k),runs["q3_ae"+k]["parameters"],min(r["val_mse"] for r in runs["q3_ae"+k]["history"]),
             v["test_mse"],v["test_probe_accuracy"]) for k,v in data["q3_probes"].items()])
        table(["CNN input","MSE to clean","Accuracy"],[(k,v["mse"],v["accuracy"]) for k,v in q3["test"].items()])
        copy = f"<p>Validation selected {q3['selected_size']} latent scalars. Denoising changed CNN test accuracy by {100*(q3['test']['denoiser']['accuracy']-q3['test']['corrupted']['accuracy']):+.2f} points relative to the noisy input.</p>"
        clean_accuracy = q3["test"]["clean"]["accuracy"]
        noisy_accuracy = q3["test"]["corrupted"]["accuracy"]
        denoised_accuracy = q3["test"]["denoiser"]["accuracy"]
        effect = ("recovered some classifier-relevant information"
                  if denoised_accuracy > noisy_accuracy else "did not improve classification")
        copy = copy[:-4] + f" The denoiser {effect} for this fixed CNN. The clean-input score was {clean_accuracy:.2%}.</p>"
        baseline = q3["test"]["clean_ae"]
        denoised = q3["test"]["denoiser"]
        if denoised["mse"] < baseline["mse"] and denoised_accuracy < baseline["accuracy"]:
            copy = copy[:-4] + (f" The denoiser lowered MSE from {baseline['mse']:.5f} to {denoised['mse']:.5f} "
                                f"relative to the clean autoencoder, while CNN accuracy fell from "
                                f"{baseline['accuracy']:.2%} to {denoised_accuracy:.2%}.</p>")
        by_mse = sorted(data["q3_probes"], key=lambda k:data["q3_probes"][k]["test_mse"])
        by_probe = sorted(data["q3_probes"], key=lambda k:-data["q3_probes"][k]["test_probe_accuracy"])
        agreement = "matched" if by_mse == by_probe else "differed"
        copy += (f"<p>The order from lowest test MSE to highest was {', '.join(by_mse)}. "
                 f"The order from highest test probe accuracy to lowest was {', '.join(by_probe)}. "
                 f"These rankings {agreement} across the tested widths.</p>")
        figure("q3_reconstructions","Clean input, noisy input, clean-autoencoder reconstruction, and denoised reconstruction, top to bottom. The denoiser keeps the broad ship and airplane silhouettes while softening fine detail.")
    else:
        copy = ""
    section(3,"Can an image look better without being easier to classify?", """
      <p>A CIFAR-10 image starts with 3,072 channel values. The autoencoders compress it into 16, 64, or 256 numbers, corresponding to 192×, 48×, and 12× fewer values. Validation pixel MSE selects the best reconstruction model. For denoising, zero-mean Gaussian noise with standard deviation 0.25 is added before clipping the image to the unit interval. Both autoencoder baselines see the same noisy test inputs, and the clean CNN from Q2 scores every output.</p>
      <p>A blurry image can have low pixel error while losing details that distinguish a cat from a dog. The linear probe checks whether class information is readily accessible from the frozen encoder. Compare the ordering of the three sizes by test MSE and probe accuracy. Agreement in this experiment would still leave open whether other architectures or nonlinear probes behave similarly.</p>
    """+copy)
    if "q4" in data:
        q4 = data["q4"]
        table(["Kernel","C","Train acc.","Val acc.","SV fraction"],[
            (r["kernel"],r["C"],r["train_accuracy"],r["val_accuracy"],r["support_fraction"]) for r in q4["sweep"]])
        table(["k","Validation silhouette"],[(k,v) for k,v in q4["silhouette"].items()])
        table(["Class"]+[str(k) for k in range(q4["selected_k"])],[
            [c]+r for c,r in zip(CLASSES,q4["contingency_class_rows"])])
        table(["Pair","Mutual errors","Train score","Test score"],[
            (" / ".join(CLASSES[i] for i in r["pair"]),r["mutual_errors"],a,b)
            for r in q4["difficult_pairs"] for a,b in zip(r["train_scores"],r["test_scores"])])
        row = max(q4["disagreement"], key=lambda r:r["cluster_cross_class_fraction"]-r["svm_mutual_error_rate"])
        names = " and ".join(CLASSES[i] for i in row["pair"])
        copy = (f"<p>The validation-selected {q4['selected_kernel']} model used C = {q4['selected_C']} "
                f"and achieved {q4['test_accuracy']:.2%} test accuracy. K-means selected k = {q4['selected_k']}. "
                f"Test ARI was {q4['test_ari']:.4f} and NMI was {q4['test_nmi']:.4f}.</p>"
                f"<p>For {names}, the mutual SVM error rate was {row['svm_mutual_error_rate']:.2%}. "
                f"Of same-cluster pairs drawn from those two classes, {row['cluster_cross_class_fraction']:.2%} crossed the class boundary.</p>")
        best_row = next(r for r in q4["sweep"] if r["kernel"] == q4["selected_kernel"] and r["C"] == q4["selected_C"])
        low_row = min((r for r in q4["sweep"] if r["kernel"] == q4["selected_kernel"]), key=lambda r:r["C"])
        high_row = max((r for r in q4["sweep"] if r["kernel"] == q4["selected_kernel"]), key=lambda r:r["C"])
        copy += (f"<p>Within the selected kernel, C = {low_row['C']} gave {low_row['train_accuracy']:.2%} training "
                 f"and {low_row['val_accuracy']:.2%} validation accuracy. At C = {high_row['C']}, these were "
                 f"{high_row['train_accuracy']:.2%} and {high_row['val_accuracy']:.2%}. The selected setting's "
                 f"training and validation gap was {100*(best_row['train_accuracy']-best_row['val_accuracy']):.2f} points.</p>")
        low_poly = min((r for r in q4["sweep"] if r["kernel"] == "poly"), key=lambda r:r["C"])
        copy = copy[:-4] + (
            f" The higher training score and lower validation score at the largest C show overfitting relative to the selected setting. "
            f"The degree-two polynomial at its smallest C underfit, with {low_poly['train_accuracy']:.2%} train "
            f"and {low_poly['val_accuracy']:.2%} validation accuracy.</p>"
            if high_row["train_accuracy"] > low_row["train_accuracy"] and high_row["val_accuracy"] < low_row["val_accuracy"] else "</p>"
        )
        copy += "".join(f"<p>{html.escape(note)}</p>" for note in observations.get("q4", []))
        for key,cap in (("val_accuracy","Validation accuracy versus C per kernel."),
                        ("support_fraction","Support-vector fraction versus C per kernel."),
                        ("boundaries","Three difficult pairs. Pair-specific support vectors appear at left and near-boundary test images at right.")):
            figure("q4_"+key, cap)
    else:
        copy = ""
    section(4,"Which classes overlap in CNN feature space?", """
      <p>The clean CNN's 216 penultimate activations stay fixed. The same 3,000 stratified training examples go into all 15 SVM fits. Compare the training-to-validation gap as C rises. A large gap can signal overfitting, while low accuracy on both sets can suggest underfitting. The fraction of support vectors records how much of the training subset remains active in the model.</p>
      <p>For each of the three most confused class pairs, inspect examples with scores near zero on that pair's decision function. A score's magnitude is a within-pair boundary proxy. K-means selects its cluster count by unlabeled validation silhouette. The test contingency table shows whether clean class separation under supervision coincides with natural clusters in the same feature space.</p>
    """+copy)
    status = "Experiment plan" if not data else ("Measured results" if "q4" in data else "Training in progress")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>What CIFAR-10 models learn from pixels</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f6f4ed;color:#192c30;font:17px/1.58 system-ui,sans-serif}
main{max-width:990px;margin:auto;padding:36px 20px 70px}header{background:#113e3b;color:white;padding:clamp(24px,5vw,55px);border-radius:18px}
h1,h2{font-family:Georgia,serif;line-height:1.1}h1{font-size:clamp(34px,7vw,65px);margin:12px 0}
h2{font-size:32px;margin:6px 0 16px}small{font-weight:bold;letter-spacing:.12em}
nav{display:flex;flex-wrap:wrap;gap:18px;margin:22px 0}a{color:#135954}
section{background:white;padding:clamp(17px,4vw,36px);margin:18px 0;border:1px solid #dcded4;border-radius:15px}
.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;margin:10px 0;width:100%}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #dcded4}th{background:#e8f3ee}
figure{margin:24px 0}.tiles{display:flex;gap:14px;flex-wrap:wrap}.tiles div{background:#e8f3ee;border-radius:8px;padding:12px 16px}.tiles pre{font-size:18px;line-height:1.3;margin:8px 0}img{max-width:100%;height:auto}figcaption,.pending{color:#496361}
.split{display:flex;height:44px;border-radius:6px;overflow:hidden;margin:14px 0}.split span{height:100%}
</style><main>""" +
    f"<header><small>{status.upper()}</small><h1>What CIFAR-10 models learn from pixels</h1>"
    f"<p>{html.escape(args.name)} · {html.escape(roll)}</p></header>"
    """<nav><a href="#q1">MLPs</a><a href="#q2">CNNs</a><a href="#q3">Autoencoders</a><a href="#q4">SVMs and clusters</a></nav>
<section><h2>One split for every experiment</h2>
<div class="split" role="img" aria-label="45,000 training, 5,000 validation, and 10,000 official test images">
<span style="width:75%;background:#23675d"></span><span style="width:8.333%;background:#dc984e"></span><span style="width:16.667%;background:#536686"></span></div>
<p>45,000 training · 5,000 validation · 10,000 official test. The split preserves class proportions. Validation selects checkpoints and hyperparameters. The official test set supplies final scores after choices are fixed.</p>
</section>""" + "".join(sections) + "</main></html>", encoding="utf-8")
    if "q4" in data:
        summary = (
            f"On the official CIFAR-10 test set, the clean CNN reached {data['q2_test']['cnn_clean']['accuracy']:.2%} "
            f"accuracy and the clean MLP reached {data['q2_test']['mlp_clean']['accuracy']:.2%}. "
            f"After retraining on pixel-shuffled inputs, the CNN reached {data['q2_test']['cnn_pixel']['accuracy']:.2%}. "
            f"The selected denoising autoencoder gave {data['q3_denoising']['test']['denoiser']['accuracy']:.2%} CNN accuracy "
            f"on restored inputs. In the frozen CNN representation, the selected SVM achieved {data['q4']['test_accuracy']:.2%} "
            f"test accuracy, while K-means gave a test adjusted Rand index of {data['q4']['test_ari']:.3f}."
        )
        (args.out.parent / "report-abstract.tex").write_text(escape_tex(summary), encoding="utf-8")
    if data:
        (args.out.parent / "report-identity.tex").write_text(
            "\\author{" + escape_tex(args.name) + r"\\" + escape_tex(roll) + r"\\"
            + (r"\url{" + args.repo + "}" if args.repo.startswith("https://") else escape_tex(args.repo)) + r"\\\texttt{" + escape_tex(args.commit) + "}}\n", encoding="utf-8")

if __name__ == "__main__":
    main()
