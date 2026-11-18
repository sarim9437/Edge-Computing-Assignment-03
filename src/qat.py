"""
qat.py
------
Question 1, Part 3: Quantization-Aware Training.

  1. initialise from the pretrained FP32 checkpoint
  2. fuse modules (QAT variant keeps BatchNorm trainable) and attach QAT qconfig
  3. prepare_qat -> inserts FakeQuantize modules for weights and activations
  4. fine-tune with fake quantization; freeze observers and BN statistics
     in the last epoch(s) so the scales settle
  5. convert to a true INT8 model, save state_dict + TorchScript checkpoint
  6. export the optimized model to ONNX for ONNX Runtime deployment
"""
import argparse
import copy
import json
import os
import time

import torch
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from torch.ao.quantization import convert, disable_observer, get_default_qat_qconfig, prepare_qat
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from common import (CKPT, INT8_STATE, ONNX_PATHS, RESULT_DIR, evaluate, get_device, get_loaders,
                    load_model, mb, set_quant_engine, set_seed, start_logging, state_dict_bytes)
from export_onnx import export_qat_onnx
from finetune import fine_tune


def to_int8(qat_model):
    """Convert a COPY of the fake-quantized model into a real INT8 model (CPU)."""
    model = copy.deepcopy(qat_model).cpu().eval()
    return convert(model, inplace=False)


def main():
    parser = argparse.ArgumentParser(description="Quantization-aware training of ResNet-18")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--freeze-observer-epoch", type=int, default=2)
    parser.add_argument("--freeze-bn-epoch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_logging("qat")
    set_seed(args.seed)
    engine = set_quant_engine()
    device = get_device()
    cpu = torch.device("cpu")
    train_loader, test_loader = get_loaders(args.batch_size, args.workers)

    # 1) Initialise from the pretrained FP32 model
    model = load_model("fp32", "cpu")
    fp32_size = mb(len(state_dict_bytes(model)))
    fp32_acc = evaluate(model, test_loader, cpu)
    print(f"FP32 starting point: accuracy {fp32_acc:.2f}% | size {fp32_size:.2f} MB\n")

    # 2) Fuse for QAT + 3) insert fake-quantization modules
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = get_default_qat_qconfig(engine, version=0)
    prepare_qat(model, inplace=True)
    n_fq = sum(isinstance(m, FakeQuantizeBase) for m in model.modules())
    print(f"[1] QAT prepared with backend '{engine}': {n_fq} FakeQuantize modules inserted. First block:")
    print(model.layers[0])
    model.to(device)

    def epoch_hook(epoch):
        if epoch >= args.freeze_observer_epoch:
            model.apply(disable_observer)
            print(f"  [epoch {epoch}] observers frozen (scales / zero-points fixed)")
        if epoch >= args.freeze_bn_epoch:
            model.apply(freeze_bn_stats)
            print(f"  [epoch {epoch}] BatchNorm statistics frozen")

    def int8_eval(m):
        return evaluate(to_int8(m), test_loader, cpu)

    # 4) Fine-tune with fake quantization (accuracy reported for the real INT8 model)
    print(f"\n[2] QAT fine-tuning for {args.epochs} epoch(s), lr {args.lr} ...")
    t0 = time.time()
    fine_tune(model, train_loader, test_loader, device, args.epochs, args.lr, tag="QAT",
              use_amp=False, epoch_hook=epoch_hook, eval_fn=int8_eval, weight_decay=5e-5)
    qat_minutes = (time.time() - t0) / 60

    # 5) ONNX export of the optimized model
    print("\n[3] Exporting QAT model to ONNX ...")
    onnx_method = export_qat_onnx(model, ONNX_PATHS["qat"])

    # 6) Convert to INT8 and save
    int8_model = to_int8(model)
    print("\n[4] Converted to INT8. First block:")
    print(int8_model.layers[0])
    torch.save(int8_model.state_dict(), INT8_STATE["qat"])
    with torch.no_grad():
        traced = torch.jit.trace(int8_model, torch.randn(1, 3, 32, 32))
    torch.jit.save(traced, CKPT["qat"])
    print(f"\n[5] Saved INT8 state_dict       -> {INT8_STATE['qat']}")
    print(f"    Saved deployable TorchScript -> {CKPT['qat']}")

    # 7) Deployment check
    deployed = load_model("qat")
    int8_acc = evaluate(deployed, test_loader, cpu)
    int8_size = mb(os.path.getsize(INT8_STATE["qat"]))

    print("\n" + "=" * 60)
    print(f"FP32 accuracy     : {fp32_acc:6.2f}%   size {fp32_size:7.2f} MB")
    print(f"QAT INT8 accuracy : {int8_acc:6.2f}%   size {int8_size:7.2f} MB")
    print(f"Accuracy change   : {int8_acc - fp32_acc:+6.2f}%   compression {fp32_size / int8_size:.2f}x")
    print(f"QAT time          : {qat_minutes:.1f} min | ONNX: {onnx_method}")
    print("=" * 60)

    with open(os.path.join(RESULT_DIR, "qat_metrics.json"), "w") as f:
        json.dump({"backend": engine, "epochs": args.epochs, "lr": args.lr,
                   "fp32_acc": fp32_acc, "int8_acc": int8_acc, "fp32_mb": fp32_size,
                   "int8_mb": int8_size, "qat_minutes": qat_minutes, "onnx_export": onnx_method},
                  f, indent=2)


if __name__ == "__main__":
    main()
