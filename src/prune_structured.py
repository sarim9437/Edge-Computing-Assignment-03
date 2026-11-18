"""
prune_structured.py
-------------------
Question 2, Part 3: structured channel pruning with architecture reconstruction.

Which channels can be removed safely?
  In a ResNet basic block the output of conv2 is ADDED to the shortcut, so block
  output channels are tied together across blocks by the residual connections.
  The channels BETWEEN conv1 and conv2 ("middle" channels) are local to a block,
  so they are pruned:
      conv1 : drop output filters          [mid, in, 3, 3]  -> [keep, in, 3, 3]
      bn1   : drop matching gamma / beta / running mean / running var -> [keep]
      conv2 : drop matching input channels [out, mid, 3, 3] -> [out, keep, 3, 3]
  All tensor dimensions stay consistent, and a NEW, physically smaller ResNet18
  is built with these widths, so inference never uses masked zero channels.

Importance criterion: L1 norm of each conv1 filter (Li et al., 2017, "Pruning
Filters for Efficient ConvNets"): small-norm filters contribute least.
"""
import argparse
import json
import os
import time

import torch

from common import (RESULT_DIR, ResNet18, count_macs, count_params, evaluate, get_device,
                    get_loaders, load_model, mb, save_float_checkpoint, set_seed, start_logging,
                    state_dict_bytes)
from finetune import fine_tune


@torch.no_grad()
def structured_channel_prune(model, ratio):
    model = model.cpu().eval()
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    new_mid, report = [], []
    for i, block in enumerate(model.layers):
        w = block.conv1.weight.detach()
        n_old = w.shape[0]
        n_keep = max(1, int(round(n_old * (1.0 - ratio))))
        importance = w.abs().sum(dim=(1, 2, 3))                       # L1 norm per filter
        keep = torch.argsort(importance, descending=True)[:n_keep].sort().values
        p = f"layers.{i}."
        sd[p + "conv1.weight"] = w[keep].clone()
        for t in ("weight", "bias", "running_mean", "running_var"):
            sd[p + "bn1." + t] = sd[p + "bn1." + t][keep].clone()
        sd[p + "conv2.weight"] = sd[p + "conv2.weight"][:, keep].clone()
        new_mid.append(n_keep)
        report.append((f"layers.{i}", n_old, n_keep, importance.min().item(),
                       importance[keep].min().item()))
    pruned = ResNet18(mid_channels=new_mid)
    pruned.load_state_dict(sd)  # strict load: every shape must match the rebuilt architecture
    return pruned, new_mid, report


def main():
    parser = argparse.ArgumentParser(description="Structured L1 channel pruning of ResNet-18")
    parser.add_argument("--ratio", type=float, default=0.5, help="fraction of middle channels removed")
    parser.add_argument("--ft-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_logging("prune_structured")
    set_seed(args.seed)
    device = get_device()
    train_loader, test_loader = get_loaders(args.batch_size, args.workers)

    original = load_model("fp32", "cpu")
    orig_acc = evaluate(original, test_loader, device)
    orig_params, orig_macs = count_params(original), count_macs(original)
    orig_mb = mb(len(state_dict_bytes(original)))
    print(f"Original FP32: accuracy {orig_acc:.2f}% | params {orig_params:,} | "
          f"MACs {orig_macs / 1e6:.1f} M | size {orig_mb:.2f} MB\n")

    t0 = time.time()
    pruned, mid, report = structured_channel_prune(original, args.ratio)
    print(f"Pruning {args.ratio * 100:.0f}% of the middle channels in every residual block:")
    print(f"{'block':10s} {'before':>7s} {'after':>7s} {'min L1 (all)':>13s} {'min L1 (kept)':>14s}")
    for name, n_old, n_keep, lo_all, lo_kept in report:
        print(f"{name:10s} {n_old:7d} {n_keep:7d} {lo_all:13.3f} {lo_kept:14.3f}")

    # Sanity check: the rebuilt network runs with valid tensor shapes
    pruned.eval()
    with torch.no_grad():
        out = pruned(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10), "invalid output shape after reconstruction"
    print("\nReconstructed architecture runs correctly (output shape (2, 10)). First block:")
    print(pruned.layers[0])

    new_params, new_macs = count_params(pruned), count_macs(pruned)
    acc_before = evaluate(pruned.to(device), test_loader, device)
    print(f"\nAfter pruning, before fine-tuning: accuracy {acc_before:.2f}%")

    acc_after = fine_tune(pruned, train_loader, test_loader, device, args.ft_epochs, args.lr,
                          tag="structured FT")
    save_float_checkpoint(pruned, "structured", test_acc=float(acc_after), prune_ratio=args.ratio)

    # Deployment check: reload the saved checkpoint from disk
    reloaded = load_model("structured", "cpu")
    acc_reload = evaluate(reloaded, test_loader, torch.device("cpu"))
    new_mb = mb(len(state_dict_bytes(reloaded)))
    minutes = (time.time() - t0) / 60

    print("\n" + "=" * 70)
    print(f"Original   : acc {orig_acc:6.2f}% | params {orig_params:>11,} | MACs {orig_macs / 1e6:7.1f} M | {orig_mb:6.2f} MB")
    print(f"Structured : acc {acc_reload:6.2f}% | params {new_params:>11,} | MACs {new_macs / 1e6:7.1f} M | {new_mb:6.2f} MB")
    print(f"Reduction  : params {100 * (1 - new_params / orig_params):5.1f}% | "
          f"MACs {100 * (1 - new_macs / orig_macs):5.1f}% | size {orig_mb / new_mb:.2f}x smaller")
    print(f"Time: {minutes:.1f} min")
    print("=" * 70)

    with open(os.path.join(RESULT_DIR, "structured_pruning_metrics.json"), "w") as f:
        json.dump({"ratio": args.ratio, "mid_channels": mid, "orig_acc": orig_acc,
                   "acc_before_ft": acc_before, "acc_after_ft": acc_reload,
                   "orig_params": orig_params, "new_params": new_params,
                   "orig_macs": orig_macs, "new_macs": new_macs,
                   "orig_mb": orig_mb, "new_mb": new_mb, "minutes": minutes}, f, indent=2)


if __name__ == "__main__":
    main()
