"""
benchmark.py
------------
Automated benchmarking on the full CIFAR-10 test set (10,000 images), on CPU.

  --suite q1  : fp32, ptq, qat                    (Question 1)
  --suite q2  : fp32, unstructured, structured    (Question 2)
  --suite all : both (default)

Per model: top-1 accuracy, checkpoint size, gzip size, compression vs FP32,
parameters, non-zero weights, sparsity, dense MACs, batch-1 latency
(mean / p50 / p95) and batch-256 throughput.

Outputs (results/):
  q1_benchmark_summary.csv  q1_test_predictions.csv
  q2_benchmark_summary.csv  q2_test_predictions.csv
  all_models_summary.csv    benchmark_comparison.png
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from common import (CKPT, CLASSES, DESCRIPTIONS, MODEL_NAMES, RESULT_DIR, ResNet18, checkpoint_bytes,
                    count_macs, count_params, gzip_mb, load_model, load_test_tensors, mb, predict,
                    print_table, start_logging, weight_stats)

SUITES = {"q1": ["fp32", "ptq", "qat"], "q2": ["fp32", "unstructured", "structured"]}
CONSOLE_COLS = ["model", "precision", "accuracy_pct", "checkpoint_mb", "gzip_mb", "compression_x",
                "gzip_compression_x", "weight_sparsity_pct", "dense_mmacs", "latency_ms_mean",
                "throughput_img_s"]


@torch.no_grad()
def latency_ms(model, xs, n, warmup=20):
    for i in range(warmup):
        model(xs[i:i + 1])
    times = []
    for i in range(n):
        x = xs[i:i + 1]
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def benchmark_model(name, xs, ys, n_latency, fp32_params):
    model = load_model(name, "cpu")
    t0 = time.perf_counter()
    logits = predict(model, xs, 256)
    total_s = time.perf_counter() - t0
    conf, pred = torch.softmax(logits, 1).max(1)
    acc = (pred == ys).float().mean().item() * 100
    lat = latency_ms(model, xs, n_latency)
    raw = checkpoint_bytes(name, model)

    if name in ("ptq", "qat"):
        precision, params, nonzero, sparsity, macs = "INT8", fp32_params, "", "", ""
    else:
        precision = "FP32"
        params = count_params(model)
        total_w, nonzero = weight_stats(model)
        sparsity = round(100.0 * (1 - nonzero / total_w), 2)
        macs = round(count_macs(model) / 1e6, 2)

    row = {
        "model": name, "description": DESCRIPTIONS[name], "precision": precision,
        "accuracy_pct": round(acc, 2),
        "checkpoint_mb": round(mb(len(raw)), 3), "gzip_mb": round(gzip_mb(raw), 3),
        "compression_x": "", "gzip_compression_x": "",
        "params": params, "nonzero_weights": nonzero, "weight_sparsity_pct": sparsity,
        "dense_mmacs": macs,
        "latency_ms_mean": round(float(lat.mean()), 3),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
        "throughput_img_s": round(len(xs) / total_s, 1),
        "test_time_s": round(total_s, 2),
    }
    return row, pred.tolist(), conf.tolist()


def write_summary(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved -> {path}")


def write_predictions(path, names, results, ys):
    header = ["index", "true_label", "true_class"]
    for n in names:
        header += [f"{n}_pred", f"{n}_class", f"{n}_confidence", f"{n}_correct"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, t in enumerate(ys.tolist()):
            row = [i, t, CLASSES[t]]
            for n in names:
                _, preds, confs = results[n]
                p = preds[i]
                row += [p, CLASSES[p], f"{confs[i]:.4f}", int(p == t)]
            writer.writerow(row)
    print(f"Saved -> {path}")


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [r["model"] for r in rows]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(names)]
    specs = [("accuracy_pct", "Top-1 accuracy (%)"), ("checkpoint_mb", "Checkpoint size (MB)"),
             ("gzip_mb", "Compressed size, gzip (MB)"), ("latency_ms_mean", "Batch-1 CPU latency (ms)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, title) in zip(axes, specs):
        vals = [r[key] for r in rows]
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        if key == "accuracy_pct":
            ax.set_ylim(min(vals) - 5, 100)
    fig.suptitle("ResNet-18 on CIFAR-10: optimization comparison (CPU)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved chart -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP32 / INT8 / pruned models on CIFAR-10")
    parser.add_argument("--suite", choices=["q1", "q2", "all"], default="all")
    parser.add_argument("--latency-samples", type=int, default=300)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    args = parser.parse_args()

    start_logging(f"benchmark_{args.suite}")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    print(f"CPU threads: {torch.get_num_threads()} | batch-1 latency samples: {args.latency_samples}\n")

    suites = ["q1", "q2"] if args.suite == "all" else [args.suite]
    needed = []
    for s in suites:
        for n in SUITES[s]:
            if n not in needed:
                needed.append(n)

    xs, ys = load_test_tensors()
    fp32_params = count_params(ResNet18())
    results = {}
    for name in needed:
        if not os.path.exists(CKPT[name]):
            print(f"[skip] {name}: checkpoint not found")
            continue
        print(f"Benchmarking {name:13s} ...")
        results[name] = benchmark_model(name, xs, ys, args.latency_samples, fp32_params)
        r = results[name][0]
        print(f"    accuracy {r['accuracy_pct']:6.2f}% | latency {r['latency_ms_mean']:7.3f} ms "
              f"| throughput {r['throughput_img_s']:8.1f} img/s")

    if "fp32" in results:
        base = results["fp32"][0]
        for row, _, _ in results.values():
            row["compression_x"] = round(base["checkpoint_mb"] / row["checkpoint_mb"], 2)
            row["gzip_compression_x"] = round(base["gzip_mb"] / row["gzip_mb"], 2)

    for s in suites:
        names = [n for n in SUITES[s] if n in results]
        if not names:
            continue
        rows = [results[n][0] for n in names]
        print(f"\n{s.upper()} benchmark (CIFAR-10 test set, 10,000 images, CPU)")
        print_table(rows, CONSOLE_COLS)
        write_summary(os.path.join(RESULT_DIR, f"{s}_benchmark_summary.csv"), rows)
        write_predictions(os.path.join(RESULT_DIR, f"{s}_test_predictions.csv"), names, results, ys)

    if args.suite == "all":
        all_names = [n for n in MODEL_NAMES if n in results]
        rows = [results[n][0] for n in all_names]
        write_summary(os.path.join(RESULT_DIR, "all_models_summary.csv"), rows)
        plot(rows, os.path.join(RESULT_DIR, "benchmark_comparison.png"))


if __name__ == "__main__":
    main()
