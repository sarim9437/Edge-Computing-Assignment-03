"""
ptq.py
------
Question 1, Part 2: Post-Training Static Quantization (PyTorch eager mode).

Pipeline:
  1. load the trained FP32 model
  2. fuse Conv+BN+ReLU and Conv+BN
  3. configure the quantization backend (fbgemm, x86 CPUs) and qconfig
  4. insert observers (prepare)
  5. calibrate on a subset of the TRAINING set
  6. convert to INT8
  7. save INT8 state_dict + deployable TorchScript checkpoint
  8. reload the saved TorchScript model and evaluate it (deployment check)
"""
import argparse
import json
import os
import time

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

from calibration import calibrate, get_calibration_loader
from common import (CKPT, INT8_STATE, RESULT_DIR, evaluate, get_test_loader, load_model, mb,
                    set_quant_engine, set_seed, start_logging, state_dict_bytes)


def main():
    parser = argparse.ArgumentParser(description="Post-training static quantization of ResNet-18")
    parser.add_argument("--calib-images", type=int, default=2000)
    parser.add_argument("--calib-batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_logging("ptq")
    set_seed(args.seed)
    engine = set_quant_engine()
    cpu = torch.device("cpu")
    test_loader = get_test_loader()

    model = load_model("fp32", "cpu")
    fp32_size = mb(len(state_dict_bytes(model)))
    fp32_acc = evaluate(model, test_loader, cpu)
    print(f"FP32 baseline: accuracy {fp32_acc:.2f}% | size {fp32_size:.2f} MB\n")

    # 1) Layer fusion (BN folded into conv weights, ReLU merged into the conv op)
    model.eval()
    model.fuse_model(is_qat=False)
    print("[1] Fusion done (Conv+BN+ReLU -> ConvReLU2d, Conv+BN -> Conv2d). First block:")
    print(model.layers[0])

    # 2) Quantization backend configuration
    model.qconfig = get_default_qconfig(engine)
    print(f"\n[2] Backend: {engine} | qconfig: {model.qconfig}")

    # 3) Insert observers, 4) calibrate
    prepare(model, inplace=True)
    print(f"\n[3] Observers inserted. Calibrating on {args.calib_images} training images ...")
    t0 = time.time()
    calibrate(model, get_calibration_loader(args.calib_images, args.calib_batch, args.seed))
    calib_time = time.time() - t0
    print(f"    calibration time: {calib_time:.1f}s")

    # 5) Convert FP32 -> INT8
    convert(model, inplace=True)
    print("\n[4] Converted to INT8. First block:")
    print(model.layers[0])

    # 6) Save: raw INT8 weights + self-contained TorchScript for deployment
    torch.save(model.state_dict(), INT8_STATE["ptq"])
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.randn(1, 3, 32, 32))
    torch.jit.save(traced, CKPT["ptq"])
    print(f"\n[5] Saved INT8 state_dict       -> {INT8_STATE['ptq']}")
    print(f"    Saved deployable TorchScript -> {CKPT['ptq']}")

    # 7) Deployment check: reload from disk and evaluate on the full test set
    deployed = load_model("ptq")
    int8_acc = evaluate(deployed, test_loader, cpu)
    int8_size = mb(os.path.getsize(INT8_STATE["ptq"]))

    print("\n" + "=" * 60)
    print(f"FP32 accuracy     : {fp32_acc:6.2f}%   size {fp32_size:7.2f} MB")
    print(f"PTQ INT8 accuracy : {int8_acc:6.2f}%   size {int8_size:7.2f} MB")
    print(f"Accuracy drop     : {fp32_acc - int8_acc:6.2f}%   compression {fp32_size / int8_size:.2f}x")
    print("=" * 60)

    with open(os.path.join(RESULT_DIR, "ptq_metrics.json"), "w") as f:
        json.dump({"backend": engine, "calibration_images": args.calib_images,
                   "calibration_seconds": calib_time, "fp32_acc": fp32_acc, "int8_acc": int8_acc,
                   "fp32_mb": fp32_size, "int8_mb": int8_size}, f, indent=2)


if __name__ == "__main__":
    main()
