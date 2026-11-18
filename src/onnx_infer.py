"""
onnx_infer.py
-------------
CPU inference + benchmarking of the exported ONNX models with ONNX Runtime
(CPUExecutionProvider). This is the deployment path: no PyTorch needed at runtime.

Examples:
  python src\onnx_infer.py                                     # benchmark all ONNX models
  python src\onnx_infer.py --image samples\test_000_cat.png    # classify one image
Output: results/onnx_benchmark.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import onnxruntime as ort

from common import (CLASSES, MODEL_NAMES, ONNX_PATHS, RESULT_DIR, load_test_tensors, mb,
                    preprocess_image, print_table, start_logging)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def make_session(path, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime CPU inference")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--latency-samples", type=int, default=300)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    start_logging("onnx_infer")
    print(f"ONNX Runtime {ort.__version__} | providers: {ort.get_available_providers()}\n")

    sessions = {}
    for name in MODEL_NAMES:
        path = ONNX_PATHS[name]
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found (run export_onnx.py / qat.py)")
            continue
        sessions[name] = make_session(path, args.threads)

    if args.image:
        x = preprocess_image(args.image).numpy()
        print(f"Image: {args.image}")
        print(f"{'model':13s} {'prediction':>11s} {'conf':>7s} {'ms':>8s}")
        for name, sess in sessions.items():
            t0 = time.perf_counter()
            logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
            ms = (time.perf_counter() - t0) * 1000
            p = softmax(logits)[0]
            k = int(p.argmax())
            print(f"{name:13s} {CLASSES[k]:>11s} {p[k] * 100:6.1f}% {ms:8.2f}")
        return

    xs, ys = load_test_tensors()
    X, Y = xs.numpy(), ys.numpy()
    rows = []
    for name, sess in sessions.items():
        inp = sess.get_inputs()[0].name
        t0 = time.perf_counter()
        preds = np.concatenate([sess.run(None, {inp: X[i:i + 256]})[0].argmax(1)
                                for i in range(0, len(X), 256)])
        total = time.perf_counter() - t0
        acc = float((preds == Y).mean() * 100)
        for i in range(20):
            sess.run(None, {inp: X[i:i + 1]})
        lat = []
        for i in range(args.latency_samples):
            t = time.perf_counter()
            sess.run(None, {inp: X[i:i + 1]})
            lat.append((time.perf_counter() - t) * 1000)
        lat = np.array(lat)
        rows.append({"model": name, "onnx_file": os.path.basename(ONNX_PATHS[name]),
                     "size_mb": round(mb(os.path.getsize(ONNX_PATHS[name])), 3),
                     "accuracy_pct": round(acc, 2),
                     "latency_ms_mean": round(float(lat.mean()), 3),
                     "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
                     "throughput_img_s": round(len(X) / total, 1)})
        print(f"  {name:13s} accuracy {acc:6.2f}% | latency {lat.mean():7.3f} ms")

    if rows:
        print("\nONNX Runtime CPU benchmark (CIFAR-10 test set, 10,000 images)")
        print_table(rows, list(rows[0].keys()))
        path = os.path.join(RESULT_DIR, "onnx_benchmark.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
