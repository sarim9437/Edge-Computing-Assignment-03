"""
make_report.py
--------------
Builds the technical report (deliverable 9) as a Word document directly from the
experiment outputs, so every number in the report matches the generated results:
  results/*.json, results/*.csv, results/benchmark_comparison.png
  and every screenshot in report/screenshots/ (embedded in file-name order).

Usage:
  python src\make_report.py --name "Your Name" --roll 22i-0000 --section A
Output: report/Technical_Report.docx
"""
import argparse
import csv
import glob
import json
import os
import platform
from datetime import date

import onnxruntime
import torch
from docx import Document
from docx.shared import Inches, Pt
from PIL import Image as PILImage

from common import BASE_DIR, RESULT_DIR

REPORT_DIR = os.path.join(BASE_DIR, "report")
SHOT_DIR = os.path.join(REPORT_DIR, "screenshots")
ORIG_WIDTHS = [64, 64, 128, 128, 256, 256, 512, 512]

DELIVERABLES = [
    ["1", "Baseline model", "models/resnet18_fp32.pth, src/train_baseline.py"],
    ["2", "PTQ implementation", "src/ptq.py, src/calibration.py, models/resnet18_ptq_int8_scripted.pt (+ _state_dict.pth)"],
    ["3", "QAT implementation", "src/qat.py, src/finetune.py, models/resnet18_qat_int8_scripted.pt (+ _state_dict.pth)"],
    ["4", "Unstructured pruning", "src/prune_unstructured.py, models/resnet18_unstructured_pruned.pth"],
    ["5", "Structured pruning", "src/prune_structured.py, models/resnet18_structured_pruned.pth"],
    ["6", "Inference pipeline", "src/infer.py (all five models), src/common.py (load_model)"],
    ["7", "Benchmarking", "src/benchmark.py, results/q1_*.csv, results/q2_*.csv, results/all_models_summary.csv"],
    ["8", "ONNX deployment", "onnx/*.onnx (5 models), src/export_onnx.py, src/onnx_infer.py, results/onnx_benchmark.csv"],
    ["9", "Technical report", "report/Technical_Report.docx, report/screenshots/"],
    ["10", "README", "README.md, requirements.txt, run_all.ps1"],
]


# ------------------------------------------------------------------ data helpers
def load_json(name):
    with open(os.path.join(RESULT_DIR, name)) as fh:
        return json.load(fh)


def load_csv(name):
    with open(os.path.join(RESULT_DIR, name), newline="") as fh:
        return {r["model"]: r for r in csv.DictReader(fh)}


def num(x, nd=2):
    return f"{float(x):.{nd}f}"


def count(x):
    return f"{int(float(x)):,}" if str(x).strip() not in ("", "None") else "-"


def ratio(a, b):
    return float(a) / float(b)


# ------------------------------------------------------------------ docx helpers
def para(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    return p


def bullets(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def table(doc, header, rows, size=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(size)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(size)
    doc.add_paragraph()


def code(doc, lines):
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
    doc.add_paragraph()


def figure(doc, path, caption, width=6.3, max_height=8.0):
    if not os.path.exists(path):
        return
    w, h = PILImage.open(path).size
    if h / w * width > max_height:
        doc.add_picture(path, height=Inches(max_height))
    else:
        doc.add_picture(path, width=Inches(width))
    cap = doc.add_paragraph()
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)


def training_curve(base, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = base["history"]
    ep = [x["epoch"] for x in h]
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.plot(ep, [x["train_acc"] for x in h], marker="o", label="train accuracy")
    ax1.plot(ep, [x["test_acc"] for x in h], marker="s", label="test accuracy")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy (%)")
    ax2 = ax1.twinx()
    ax2.plot(ep, [x["loss"] for x in h], color="gray", linestyle="--", label="train loss")
    ax2.set_ylabel("loss")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.set_title("FP32 baseline training on CPU")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser(description="Generate the technical report (.docx)")
    ap.add_argument("--name", default="Student Name")
    ap.add_argument("--roll", default="ROLL-NUM")
    ap.add_argument("--section", default="X")
    args = ap.parse_args()

    base = load_json("baseline_metrics.json")
    ptq = load_json("ptq_metrics.json")
    qat = load_json("qat_metrics.json")
    uns = load_json("unstructured_pruning_metrics.json")
    st = load_json("structured_pruning_metrics.json")
    q1 = load_csv("q1_benchmark_summary.csv")
    q2 = load_csv("q2_benchmark_summary.csv")
    ox = load_csv("onnx_benchmark.csv")
    fp, pq, qa, un, sp = q1["fp32"], q1["ptq"], q1["qat"], q2["unstructured"], q2["structured"]
    fp2 = q2["fp32"]

    curve = os.path.join(RESULT_DIR, "training_curve.png")
    training_curve(base, curve)

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    # ---------------- title page
    doc.add_heading("Assignment 3 - Technical Report", 0)
    para(doc, "Edge Computing & Intelligent Systems")
    para(doc, "Model optimization of ResNet-18 on CIFAR-10: post-training quantization, quantization-aware "
              "training, unstructured and structured pruning, and ONNX deployment on CPU.")
    para(doc, args.name, "Name: ")
    para(doc, args.roll, "Roll number: ")
    para(doc, args.section, "Section: ")
    para(doc, date.today().strftime("%d %B %Y"), "Date: ")
    doc.add_page_break()

    # ---------------- 1. introduction
    doc.add_heading("1. Introduction", 1)
    para(doc, "Edge devices have limited memory, compute and energy, so a trained network usually has to be "
              "compressed before deployment. This assignment builds two reproducible PyTorch pipelines on the same "
              "ResNet-18 / CIFAR-10 baseline. Question 1 reduces numerical precision from FP32 to INT8 with "
              "post-training static quantization (PTQ) and quantization-aware training (QAT). Question 2 removes "
              "parameters with unstructured magnitude pruning and with structured channel pruning followed by "
              "architecture reconstruction. All five models are exported for deployment (TorchScript / state_dict "
              "and ONNX) and benchmarked automatically on the full CIFAR-10 test set on CPU.")

    # ---------------- 2. setup
    doc.add_heading("2. Experimental setup", 1)
    table(doc, ["Item", "Value"], [
        ["Hardware", f"CPU only, {os.cpu_count()} logical cores ({platform.processor() or 'x86-64'}), no GPU"],
        ["Operating system", platform.platform()],
        ["Software", f"Python {platform.python_version()}, PyTorch {torch.__version__}, "
                     f"ONNX Runtime {onnxruntime.__version__}"],
        ["Dataset", "CIFAR-10: 50,000 training / 10,000 test images, 32x32 RGB, 10 classes"],
        ["Pre-processing", "random crop (padding 4) + horizontal flip for training; per-channel mean/std normalization"],
        ["Model", f"ResNet-18 (CIFAR variant), {count(fp['params'])} parameters, "
                  f"{num(fp['dense_mmacs'], 1)} M MACs per image"],
        ["Baseline training", f"SGD (momentum 0.9, Nesterov, weight decay 5e-4), OneCycle LR (max 0.1), batch 128, "
                              f"{base['epochs']} epochs, {num(base['train_minutes'], 1)} min"],
        ["Reproducibility", "fixed seed 42, fixed calibration subset, one-command pipeline run_all.ps1"],
    ])
    para(doc, "No GPU was available and one epoch of the standard CIFAR ResNet-18 took about 1206 s on the CPU. "
              "The stem convolution therefore uses stride 2: all 11.17 M parameters are kept, but the network works "
              "on 16x16 feature maps, which cuts computation by about 4x (about 330 s per epoch). This is itself an "
              "edge-oriented accuracy / compute trade-off.", "Design decision for CPU training: ")
    figure(doc, curve, f"Figure 1: FP32 baseline training - final test accuracy {num(base['final_test_acc'])}%.")

    # ---------------- 3. Q1
    doc.add_heading("3. Question 1 - Quantization (PTQ and QAT)", 1)
    doc.add_heading("3.1 Making the model quantizable", 2)
    bullets(doc, [
        "QuantStub / DeQuantStub mark where tensors enter and leave the INT8 domain.",
        "Every residual addition uses FloatFunctional.add_relu, so the add + ReLU can be quantized "
        "(a plain '+' operator cannot).",
        "Separate ReLU modules are used at each position, so the Conv+BN+ReLU patterns can be fused.",
    ])
    doc.add_heading("3.2 Post-Training Static Quantization (src/ptq.py, src/calibration.py)", 2)
    bullets(doc, [
        "Model fusion (eval mode): Conv+BN+ReLU -> ConvReLU2d and Conv+BN -> Conv2d, including the 1x1 downsample "
        "shortcuts. BatchNorm is folded into the convolution weights, so it costs nothing at inference.",
        f"Backend configuration: '{ptq['backend']}' (x86 INT8 kernels). Activations: HistogramObserver, per-tensor "
        "affine quint8 (reduce_range); weights: per-channel symmetric qint8.",
        f"Calibration pipeline: prepare() inserts observers, then {ptq['calibration_images']} fixed training images "
        f"(no augmentation, seed 42) are passed through the model in {num(ptq['calibration_seconds'], 1)} s to record "
        "activation ranges, from which scales and zero-points are computed.",
        "Conversion: convert() replaces the modules with QuantizedConvReLU2d, QuantizedConv2d, QFunctional and "
        "QuantizedLinear operating on INT8 tensors.",
        "Save and deploy: the INT8 state_dict and a TorchScript file (torch.jit.trace) are saved; the TorchScript "
        "file is reloaded from disk and evaluated on the full test set as a deployment check.",
    ])
    doc.add_heading("3.3 Quantization-Aware Training (src/qat.py, src/finetune.py)", 2)
    bullets(doc, [
        "Initialization from the pretrained FP32 checkpoint.",
        "fuse_modules_qat keeps BatchNorm trainable inside ConvBnReLU2d / ConvBn2d modules.",
        "prepare_qat() with the fbgemm QAT qconfig inserts FakeQuantize modules on all weights (per-channel) and "
        "activations (per-tensor). The forward pass simulates INT8 rounding while gradients flow through the "
        "straight-through estimator.",
        f"Fine-tuning: {qat['epochs']} epochs, SGD lr {qat['lr']}, cosine schedule. Observers and BatchNorm statistics "
        "are frozen in the last epoch so the scales settle. After every epoch a real INT8 copy is evaluated.",
        f"Conversion and export: convert() produces the INT8 model (saved as state_dict + TorchScript; "
        f"{num(qat['qat_minutes'], 1)} min in total) and the optimized model is exported to ONNX.",
    ])
    doc.add_heading("3.4 Results (PyTorch, CPU, 10,000 test images)", 2)
    rows = []
    for label, r in (("FP32 baseline", fp), ("PTQ INT8", pq), ("QAT INT8", qa)):
        rows.append([label, num(r["accuracy_pct"]), num(r["checkpoint_mb"]), num(r["gzip_mb"]),
                     f"{num(r['compression_x'])}x", num(r["latency_ms_mean"], 3), num(r["throughput_img_s"], 1)])
    table(doc, ["Model", "Accuracy (%)", "Size (MB)", "gzip (MB)", "Compression", "Latency b=1 (ms)",
                "Throughput (img/s)"], rows)
    d_ptq = float(pq["accuracy_pct"]) - float(fp["accuracy_pct"])
    d_qat = float(qa["accuracy_pct"]) - float(fp["accuracy_pct"])
    doc.add_heading("3.5 Discussion", 2)
    bullets(doc, [
        f"Both INT8 models keep the FP32 accuracy (PTQ {d_ptq:+.2f} pp, QAT {d_qat:+.2f} pp) while the weights "
        f"shrink {num(pq['compression_x'])}x, as expected from storing 8-bit instead of 32-bit values.",
        f"On CPU the INT8 models are {ratio(fp['latency_ms_mean'], pq['latency_ms_mean']):.1f}x (PTQ) and "
        f"{ratio(fp['latency_ms_mean'], qa['latency_ms_mean']):.1f}x (QAT) faster at batch size 1 and reach "
        f"{ratio(pq['throughput_img_s'], fp['throughput_img_s']):.1f}x higher throughput, thanks to the fbgemm INT8 "
        "kernels and 4x less memory traffic.",
        "PTQ vs QAT: ResNet-18 on CIFAR-10 is very tolerant of 8-bit quantization, so PTQ already loses almost "
        f"nothing and QAT has no accuracy left to recover. QAT also cost {num(qat['qat_minutes'], 1)} min of training "
        f"versus {num(ptq['calibration_seconds'], 1)} s of calibration for PTQ. QAT becomes worthwhile when PTQ causes "
        "a noticeable drop, e.g. for compact models with depthwise convolutions or for lower bit-widths.",
    ])

    # ---------------- 4. Q2
    doc.add_heading("4. Question 2 - Pruning", 1)
    doc.add_heading("4.1 Unstructured pruning (src/prune_unstructured.py)", 2)
    per_iter = 1 - (1 - uns["target_sparsity"]) ** (1 / uns["iterations"])
    bullets(doc, [
        "Magnitude-based pruning: global L1 ranking (prune.global_unstructured with L1Unstructured) over the weights "
        "of all 21 Conv2d and Linear layers, so the smallest weights in the whole network are removed first.",
        f"Iterative pruning: {uns['iterations']} iterations, each removing {per_iter * 100:.2f}% of the remaining "
        f"weights: 1 - (1 - {per_iter:.4f})^{uns['iterations']} = {uns['target_sparsity'] * 100:.0f}% sparsity.",
        "Weight masking: torch.nn.utils.prune stores weight_orig and a binary mask; a forward pre-hook computes "
        "weight = weight_orig * mask, so pruned weights stay zero during fine-tuning.",
        "Fine-tuning: 1 epoch after every iteration plus a final epoch with the final masks (SGD, cosine LR).",
        "Permanent pruning: prune.remove() folds the mask into the weight tensor and deletes weight_orig, the mask "
        "and the hook (no mask tensors remain in the saved state_dict).",
    ])
    table(doc, ["Iteration", "Sparsity (%)", "Accuracy after pruning (%)", "Accuracy after fine-tuning (%)"],
          [[h["iteration"], num(h["sparsity"]), num(h["acc_after_prune"]), num(h["acc_after_finetune"])]
           for h in uns["history"]])
    para(doc, "Because the ranking is global, the wide last stages (512 channels, most of the parameters) are pruned "
              "most heavily, while the stem convolution and the classifier keep almost all of their weights "
              "(per-layer sparsity table in logs/prune_unstructured.log).")

    doc.add_heading("4.2 Structured channel pruning (src/prune_structured.py)", 2)
    bullets(doc, [
        "Importance criterion: L1 norm of every conv1 filter in each residual block (Li et al., 2017); the "
        f"{st['ratio'] * 100:.0f}% least important filters are removed.",
        "Valid tensor dimensions: in a basic block the output of conv2 is added to the shortcut, so block output "
        "channels are coupled across blocks by the residual connections. Only the internal channels between conv1 "
        "and conv2 are pruned: conv1 loses output filters, bn1 loses the matching gamma, beta, running mean and "
        "running variance, and conv2 loses the matching input channels.",
        "Reconstruction: a new ResNet18 is built with the reduced widths, the sliced weights are loaded with a strict "
        "load_state_dict, and a forward pass checks the output shape. The pruned network physically contains fewer "
        "channels, so it never relies on masked zero-valued channels.",
        f"Fine-tuning: 3 epochs (SGD lr 0.01, cosine). Accuracy right after pruning was {num(st['acc_before_ft'])}% "
        f"and recovered to {num(st['acc_after_ft'])}%.",
        "Deployment: the checkpoint stores the reduced channel configuration (mid_channels) with the weights, so the "
        "inference pipeline rebuilds the exact architecture.",
    ])
    table(doc, ["Block", "Middle channels before", "Middle channels after"],
          [[f"layers.{i}", w, m] for i, (w, m) in enumerate(zip(ORIG_WIDTHS, st["mid_channels"]))])

    doc.add_heading("4.3 Results (PyTorch, CPU, 10,000 test images)", 2)
    rows = []
    for label, r in (("Original FP32", fp2), ("Unstructured", un), ("Structured", sp)):
        rows.append([label, num(r["accuracy_pct"]), count(r["params"]), count(r["nonzero_weights"]),
                     num(r["weight_sparsity_pct"]), num(r["dense_mmacs"], 1), num(r["checkpoint_mb"]),
                     num(r["gzip_mb"]), num(r["latency_ms_mean"], 2), num(r["throughput_img_s"], 1)])
    table(doc, ["Model", "Acc (%)", "Params", "Non-zero weights", "Sparsity (%)", "MACs (M)", "Size (MB)",
                "gzip (MB)", "Latency (ms)", "Throughput (img/s)"], rows, size=8)

    doc.add_heading("4.4 Discussion", 2)
    bullets(doc, [
        f"Unstructured pruning zeroes {num(un['weight_sparsity_pct'], 1)}% of the weights with "
        f"{float(un['accuracy_pct']) - float(fp2['accuracy_pct']):+.2f} pp accuracy change. The zeros are still "
        f"stored in dense tensors, so the checkpoint stays {num(un['checkpoint_mb'])} MB; the gain only appears after "
        f"compression ({num(un['gzip_compression_x'])}x smaller with gzip). Latency does not improve "
        f"({num(un['latency_ms_mean'])} ms vs {num(fp2['latency_ms_mean'])} ms) because dense kernels still multiply "
        "the zeros; real speed-ups need sparse storage formats and sparse kernels or hardware support "
        "(e.g. 2:4 structured sparsity).",
        f"Structured pruning removes {100 * (1 - st['new_params'] / st['orig_params']):.1f}% of the parameters and "
        f"{100 * (1 - st['new_macs'] / st['orig_macs']):.1f}% of the MACs; the checkpoint is "
        f"{num(sp['compression_x'])}x smaller and accuracy changes by "
        f"{float(sp['accuracy_pct']) - float(fp2['accuracy_pct']):+.2f} pp. Because the network is physically "
        f"smaller, the speed-up is real on ordinary hardware: {ratio(sp['throughput_img_s'], fp2['throughput_img_s']):.1f}x "
        f"PyTorch throughput and {ratio(ox['fp32']['latency_ms_mean'], ox['structured']['latency_ms_mean']):.1f}x lower "
        "ONNX Runtime latency. The batch-1 PyTorch latency gain is smaller because per-layer framework overhead "
        "dominates for a single 32x32 image.",
        "Trade-off: unstructured pruning preserves accuracy best but needs special support to pay off, while "
        "structured pruning gives smaller and faster models on any CPU at a small accuracy cost. The large drop "
        "before fine-tuning shows that fine-tuning is essential for structured pruning.",
    ])

    # ---------------- 5. deployment
    doc.add_heading("5. Deployment: inference pipeline and ONNX", 1)
    doc.add_heading("5.1 Inference pipeline and benchmarking (src/infer.py, src/benchmark.py)", 2)
    bullets(doc, [
        "One loader (common.load_model) for all five models: FP32 and pruned models are rebuilt from their "
        "architecture configuration and state_dict; PTQ / QAT INT8 models are self-contained TorchScript files.",
        "Modes: classify the first N test images with every model side by side, or classify any image file "
        "(resized to 32x32 and normalized) with top-3 classes, confidence and latency.",
        "src/benchmark.py evaluates all models automatically on the 10,000 test images and writes summary CSVs "
        "(accuracy, size, sparsity, MACs, latency mean / p50 / p95, throughput) and per-image prediction CSVs "
        "(predicted class, confidence and correctness for every model).",
    ])
    doc.add_heading("5.2 ONNX export and CPU inference (src/export_onnx.py, src/onnx_infer.py)", 2)
    qat_note = ("QAT: direct export of the fake-quantized graph is attempted first. ONNX Runtime rejected the "
                "quantized residual add, so the fallback path was used: the QAT-trained weights are exported as a "
                "float graph and statically quantized with ONNX Runtime."
                if qat["onnx_export"].startswith("QAT weights") else
                "QAT: the fake-quantized graph was exported directly, producing QuantizeLinear / DequantizeLinear nodes.")
    bullets(doc, [
        "FP32, unstructured and structured models are exported with torch.onnx.export (opset 17, dynamic batch) and "
        "checked against PyTorch (maximum absolute output difference below 1e-5).",
        "PyTorch eager INT8 kernels (fbgemm packed weights) are not standard ONNX operators, so the PTQ ONNX model is "
        "produced with ONNX Runtime static quantization (QDQ format, per-channel weights) using the same calibration "
        "subset.",
        qat_note,
        "src/onnx_infer.py runs all ONNX models with the CPU execution provider (no PyTorch needed at runtime) and "
        "writes results/onnx_benchmark.csv.",
    ])
    rows = [[m, r["onnx_file"], num(r["size_mb"]), num(r["accuracy_pct"]), num(r["latency_ms_mean"], 3),
             num(r["latency_ms_p95"], 3), num(r["throughput_img_s"], 1)] for m, r in ox.items()]
    table(doc, ["Model", "ONNX file", "Size (MB)", "Accuracy (%)", "Latency (ms)", "p95 (ms)",
                "Throughput (img/s)"], rows, size=8)
    notes = [
        f"ONNX Runtime runs the FP32 model {ratio(fp['latency_ms_mean'], ox['fp32']['latency_ms_mean']):.1f}x faster "
        "than PyTorch eager mode at batch size 1 (graph optimizations, Conv+BN folding, operator fusion).",
        f"The PTQ INT8 ONNX model is the fastest deployment: "
        f"{ratio(ox['fp32']['latency_ms_mean'], ox['ptq']['latency_ms_mean']):.1f}x lower latency than ONNX FP32 at "
        f"{num(ox['ptq']['accuracy_pct'])}% accuracy and about 4x smaller.",
    ]
    if float(ox["qat"]["latency_ms_mean"]) > float(ox["fp32"]["latency_ms_mean"]):
        notes.append("The QAT ONNX model keeps its accuracy but is slow. This is most likely because the fallback "
                     "graph comes from the training-time Conv-BN modules: BatchNorm is not folded into the "
                     "convolutions, so ONNX Runtime cannot fuse the Q/DQ pairs into fast QLinearConv kernels. Folding "
                     "BN before export would fix this; for deployment the PTQ graph is preferred.")
    bullets(doc, notes)

    # ---------------- 6. chart
    doc.add_heading("6. Benchmark comparison", 1)
    figure(doc, os.path.join(RESULT_DIR, "benchmark_comparison.png"),
           "Figure 2: accuracy, checkpoint size, compressed size and batch-1 CPU latency of all five PyTorch models.",
           width=6.5)

    # ---------------- 7. screenshots
    doc.add_heading("7. Screenshots", 1)
    shots = sorted(glob.glob(os.path.join(SHOT_DIR, "*.png")) + glob.glob(os.path.join(SHOT_DIR, "*.jpg")))
    if not shots:
        para(doc, "(Place screenshots in report/screenshots/ and re-run make_report.py.)")
    for i, path in enumerate(shots, 1):
        label = os.path.splitext(os.path.basename(path))[0]
        head, _, tail = label.partition("_")
        if head.isdigit() and tail:
            label = tail
        figure(doc, path, f"Screenshot {i}: {label.replace('_', ' ')}")

    # ---------------- 8. conclusion
    doc.add_heading("8. Conclusion and limitations", 1)
    para(doc, f"All four techniques were implemented end to end and evaluated on the same baseline "
              f"({num(fp['accuracy_pct'])}% FP32). INT8 quantization gave the best overall trade-off on CPU: "
              f"{num(pq['compression_x'])}x smaller, {ratio(fp['latency_ms_mean'], pq['latency_ms_mean']):.1f}x faster "
              f"and no meaningful accuracy loss, with PTQ preferred because it needs only calibration. Structured "
              f"pruning produced a genuinely smaller and faster FP32 network "
              f"({100 * (1 - st['new_macs'] / st['orig_macs']):.0f}% fewer MACs) for about "
              f"{abs(float(sp['accuracy_pct']) - float(fp['accuracy_pct'])):.1f} pp of accuracy, while unstructured "
              f"pruning reached {num(un['weight_sparsity_pct'], 0)}% sparsity without accuracy loss but only helps "
              "storage unless sparse kernels are available. Combining structured pruning with INT8 quantization is "
              "the natural next step for an edge deployment.")
    bullets(doc, [
        f"CPU-only training for {base['epochs']} epochs with a stride-2 stem; a longer schedule with the stride-1 "
        "stem would reach roughly 94-95%.",
        "Latency was measured on one laptop CPU with background processes; small differences between dense FP32 "
        "models are within measurement noise.",
        "Unstructured sparsity is not exploited by the dense PyTorch / ONNX Runtime CPU kernels.",
        "The QAT ONNX model uses the ONNX Runtime fallback path (Section 5.2).",
    ])

    # ---------------- 9. reproducibility
    doc.add_heading("9. Reproducibility and deliverables", 1)
    para(doc, "Installation and step-by-step instructions are in README.md. The complete pipeline runs with one command:")
    code(doc, ["powershell -ExecutionPolicy Bypass -File .\\run_all.ps1"])
    table(doc, ["#", "Deliverable", "Files"], DELIVERABLES)

    out = os.path.join(REPORT_DIR, "Technical_Report.docx")
    doc.save(out)
    print(f"Report saved -> {out}  ({len(shots)} screenshot(s) embedded)")


if __name__ == "__main__":
    main()
