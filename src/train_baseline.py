"""
train_baseline.py
-----------------
Step 1 (Q1 and Q2): train the FP32 ResNet-18 baseline on CIFAR-10.
SGD + Nesterov momentum, OneCycle LR, mixed precision on GPU.

Outputs: models/resnet18_fp32.pth, results/baseline_metrics.json, logs/train_baseline.log
"""
import argparse
import json
import os
import time

import torch

from common import (CKPT, RESULT_DIR, ResNet18, count_params, evaluate, get_device, get_loaders,
                    save_float_checkpoint, set_seed, start_logging, train_one_epoch)


def main():
    parser = argparse.ArgumentParser(description="Train FP32 ResNet-18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_logging("train_baseline")
    set_seed(args.seed)
    device = get_device()
    train_loader, test_loader = get_loaders(args.batch_size, args.workers)

    model = ResNet18().to(device)
    print(f"Model: ResNet-18 (CIFAR variant) | parameters: {count_params(model):,}")
    print(f"Epochs: {args.epochs} | batch: {args.batch_size} | max LR: {args.lr}\n")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    use_amp = device.type == "cuda"

    history, start = [], time.time()
    test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler, use_amp)
        test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "loss": loss, "train_acc": train_acc, "test_acc": test_acc})
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {loss:.4f} | train {train_acc:6.2f}% "
              f"| test {test_acc:6.2f}% | {dt:6.1f}s")

    save_float_checkpoint(model, "fp32", test_acc=float(test_acc), epochs=args.epochs)
    total_min = (time.time() - start) / 60
    with open(os.path.join(RESULT_DIR, "baseline_metrics.json"), "w") as f:
        json.dump({"final_test_acc": test_acc, "epochs": args.epochs,
                   "train_minutes": total_min, "history": history}, f, indent=2)

    print(f"\nFinal FP32 test accuracy: {test_acc:.2f}% | training time {total_min:.1f} min")
    print(f"Saved checkpoint -> {CKPT['fp32']}")


if __name__ == "__main__":
    main()
