"""
prune_unstructured.py
---------------------
Question 2, Part 2: unstructured (fine-grained) pruning.

  * global magnitude (L1) pruning over ALL Conv2d and Linear weights
  * iterative schedule: each iteration removes the same fraction of the
    REMAINING weights, so total sparsity reaches the target after k iterations
      per_iteration = 1 - (1 - target) ** (1 / k)
  * weight masks (torch.nn.utils.prune) keep pruned weights at zero while fine-tuning
  * fine-tuning after every iteration, then a short final fine-tune
  * prune.remove() makes pruning permanent (mask folded into the weight tensor)
"""
import argparse
import json
import os
import time

import torch.nn as nn
import torch.nn.utils.prune as prune

from common import (RESULT_DIR, evaluate, get_device, get_loaders, gzip_mb, load_model, mb,
                    save_float_checkpoint, set_seed, start_logging, state_dict_bytes)
from finetune import fine_tune


def prunable_params(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def global_sparsity(params):
    zeros = sum(int((m.weight == 0).sum()) for m, _ in params)
    total = sum(m.weight.numel() for m, _ in params)
    return 100.0 * zeros / total


def main():
    parser = argparse.ArgumentParser(description="Iterative unstructured magnitude pruning")
    parser.add_argument("--target-sparsity", type=float, default=0.8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--ft-epochs", type=int, default=1, help="fine-tune epochs after each iteration")
    parser.add_argument("--final-ft-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_logging("prune_unstructured")
    set_seed(args.seed)
    device = get_device()
    train_loader, test_loader = get_loaders(args.batch_size, args.workers)

    model = load_model("fp32", device)
    base_bytes = state_dict_bytes(model)
    base_acc = evaluate(model, test_loader, device)
    print(f"Original FP32: accuracy {base_acc:.2f}% | size {mb(len(base_bytes)):.2f} MB "
          f"| gzip {gzip_mb(base_bytes):.2f} MB\n")

    params = prunable_params(model)
    per_iter = 1.0 - (1.0 - args.target_sparsity) ** (1.0 / args.iterations)
    print(f"Target sparsity {args.target_sparsity * 100:.0f}% in {args.iterations} iterations "
          f"-> prune {per_iter * 100:.2f}% of the remaining weights per iteration "
          f"({len(params)} Conv/Linear layers, global ranking)")

    history, t0 = [], time.time()
    for it in range(1, args.iterations + 1):
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=per_iter)
        sp = global_sparsity(params)
        acc_pruned = evaluate(model, test_loader, device)
        print(f"\n[iteration {it}/{args.iterations}] sparsity {sp:6.2f}% "
              f"| accuracy right after pruning {acc_pruned:6.2f}%")
        acc_ft = fine_tune(model, train_loader, test_loader, device, args.ft_epochs, args.lr,
                           tag=f"iter {it} FT")
        history.append({"iteration": it, "sparsity": sp, "acc_after_prune": acc_pruned,
                        "acc_after_finetune": acc_ft})

    if args.final_ft_epochs > 0:
        print("\n[final] fine-tuning with the final masks ...")
        fine_tune(model, train_loader, test_loader, device, args.final_ft_epochs, args.lr / 2,
                  tag="final FT")

    # Make pruning permanent: weight = weight_orig * mask, then drop mask + hook
    for module, name in params:
        prune.remove(module, name)
    leftover = [k for k in model.state_dict() if k.endswith("_orig") or k.endswith("_mask")]
    print(f"\nPruning made permanent (remaining mask tensors in state_dict: {len(leftover)})")

    final_sp = global_sparsity(params)
    final_acc = evaluate(model, test_loader, device)
    minutes = (time.time() - t0) / 60

    print(f"\n{'layer':28s} {'weight shape':>22s} {'sparsity':>9s}")
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            s = 100.0 * float((m.weight == 0).sum()) / m.weight.numel()
            print(f"{name:28s} {str(tuple(m.weight.shape)):>22s} {s:8.2f}%")

    save_float_checkpoint(model, "unstructured", test_acc=float(final_acc), sparsity=float(final_sp))
    pr_bytes = state_dict_bytes(model)

    print("\n" + "=" * 66)
    print(f"Original     : acc {base_acc:6.2f}% | dense {mb(len(base_bytes)):6.2f} MB | gzip {gzip_mb(base_bytes):6.2f} MB")
    print(f"Unstructured : acc {final_acc:6.2f}% | dense {mb(len(pr_bytes)):6.2f} MB | gzip {gzip_mb(pr_bytes):6.2f} MB")
    print(f"Global weight sparsity: {final_sp:.2f}% | time {minutes:.1f} min")
    print("Note: zeros are stored densely, so the gain appears in compressed size;")
    print("      speed-ups need sparse kernels / hardware support.")
    print("=" * 66)

    with open(os.path.join(RESULT_DIR, "unstructured_pruning_metrics.json"), "w") as f:
        json.dump({"target_sparsity": args.target_sparsity, "iterations": args.iterations,
                   "base_acc": base_acc, "final_acc": final_acc, "final_sparsity": final_sp,
                   "minutes": minutes, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
