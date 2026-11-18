"""
infer.py
--------
Inference pipeline for all five deployable PyTorch models:
  fp32 | ptq (INT8) | qat (INT8) | unstructured (pruned) | structured (pruned)

Examples:
  python src\infer.py --model all --num 10                     # first 10 CIFAR-10 test images
  python src\infer.py --model ptq --image samples\test_000_cat.png
  python src\infer.py --save-samples 3                         # save + classify sample PNGs
"""
import argparse
import os
import time

import torch
import torchvision

from common import (CLASSES, DATA_DIR, DESCRIPTIONS, MODEL_NAMES, SAMPLE_DIR, get_test_set,
                    load_model, preprocess_image, start_logging)


def save_samples(n):
    raw = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
    paths = []
    for i in range(n):
        img, label = raw[i]
        path = os.path.join(SAMPLE_DIR, f"test_{i:03d}_{CLASSES[label]}.png")
        img.save(path)
        paths.append(path)
    print(f"Saved {n} sample image(s) to {SAMPLE_DIR}")
    return paths


@torch.no_grad()
def classify_image(models, path):
    x = preprocess_image(path)
    print(f"\nImage: {path}")
    print(f"{'model':13s} {'prediction':>11s} {'conf':>7s}   {'top-3':40s} {'ms':>7s}")
    for name, model in models.items():
        t0 = time.perf_counter()
        probs = torch.softmax(model(x).float(), dim=1)[0]
        ms = (time.perf_counter() - t0) * 1000
        top = torch.topk(probs, 3)
        idx, val = top.indices.tolist(), top.values.tolist()
        top3 = ", ".join(f"{CLASSES[i]} {p * 100:.1f}%" for i, p in zip(idx, val))
        print(f"{name:13s} {CLASSES[idx[0]]:>11s} {val[0] * 100:6.1f}%   {top3:40s} {ms:7.2f}")


@torch.no_grad()
def classify_test_images(models, n):
    test_set = get_test_set()
    xs = torch.stack([test_set[i][0] for i in range(n)])
    ys = [test_set[i][1] for i in range(n)]
    preds = {name: model(xs).float().argmax(1).tolist() for name, model in models.items()}

    header = f"{'idx':>4s} {'true':>11s}" + "".join(f" {name:>13s}" for name in models)
    print("\n" + header)
    print("-" * len(header))
    for i in range(n):
        row = f"{i:4d} {CLASSES[ys[i]]:>11s}"
        for name in models:
            p = preds[name][i]
            row += f" {CLASSES[p] + ('' if p == ys[i] else '*'):>13s}"
        print(row)
    print("-" * len(header))
    acc = f"{'':4s} {'correct':>11s}"
    for name in models:
        c = sum(int(p == y) for p, y in zip(preds[name], ys))
        acc += f" {f'{c}/{n}':>13s}"
    print(acc)
    print("(* = wrong prediction)")


def main():
    parser = argparse.ArgumentParser(description="Inference with FP32 / INT8 / pruned ResNet-18 models")
    parser.add_argument("--model", choices=MODEL_NAMES + ["all"], default="all")
    parser.add_argument("--image", type=str, default=None, help="path to an image file")
    parser.add_argument("--num", type=int, default=10, help="number of test images to classify")
    parser.add_argument("--save-samples", type=int, default=0, help="save N test images as PNG and classify them")
    args = parser.parse_args()

    start_logging("infer")
    names = MODEL_NAMES if args.model == "all" else [args.model]
    models = {}
    for name in names:
        try:
            models[name] = load_model(name, "cpu")
            print(f"Loaded {name:13s} - {DESCRIPTIONS[name]}")
        except FileNotFoundError as e:
            print(f"[skip] {e}")
    if not models:
        return

    if args.image:
        classify_image(models, args.image)
    elif args.save_samples > 0:
        for path in save_samples(args.save_samples):
            classify_image(models, path)
    else:
        classify_test_images(models, args.num)


if __name__ == "__main__":
    main()
