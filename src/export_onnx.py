"""
export_onnx.py
--------------
ONNX deployment (deliverable 8).

  * FP32, unstructured-pruned and structured-pruned models: exported directly
    from PyTorch (opset 17, dynamic batch) and verified against PyTorch outputs.
  * PTQ INT8: PyTorch eager INT8 kernels (fbgemm packed weights) are not standard
    ONNX ops, so the FP32 graph is statically quantized with ONNX Runtime
    (QDQ format, per-channel weights) using the SAME calibration subset.
  * QAT INT8: exported by qat.py through export_qat_onnx() below: the
    fake-quantized model becomes QuantizeLinear/DequantizeLinear nodes.

Run:  python src\export_onnx.py
"""
import argparse
import copy
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from torch.ao.quantization import disable_fake_quant, disable_observer

from calibration import get_calibration_loader
from common import CKPT, ONNX_PATHS, load_model, load_test_tensors, mb, set_seed, start_logging

OPSET = 17


def export_float(model, path):
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy, path, opset_version=OPSET, do_constant_folding=True,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    onnx.checker.check_model(onnx.load(path))


class CifarCalibrationReader(CalibrationDataReader):
    """Feeds the calibration subset to ONNX Runtime's static quantizer."""

    def __init__(self, num_images=1000, batch_size=50):
        loader = get_calibration_loader(num_images, batch_size)
        self.batches = [{"input": x.numpy()} for x, _ in loader]
        self.it = iter(self.batches)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.batches)


def ort_static_quantize(fp32_path, int8_path, num_images=1000):
    quantize_static(fp32_path, int8_path, CifarCalibrationReader(num_images),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)


def ort_session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@torch.no_grad()
def max_abs_diff(torch_model, onnx_path, xs):
    ref = torch_model(xs).float().numpy()
    out = ort_session(onnx_path).run(None, {"input": xs.numpy()})[0]
    return float(np.abs(ref - out).max())


def top1_agreement(path_a, path_b, xs):
    a = ort_session(path_a).run(None, {"input": xs.numpy()})[0].argmax(1)
    b = ort_session(path_b).run(None, {"input": xs.numpy()})[0].argmax(1)
    return float((a == b).mean() * 100)


def export_qat_onnx(qat_model, path):
    """Export a fake-quantized (QAT) model to ONNX. Falls back to ORT static quantization."""
    m = copy.deepcopy(qat_model).cpu().eval()
    m.apply(disable_observer)
    dummy = torch.randn(1, 3, 32, 32)
    try:
        torch.onnx.export(m, dummy, path, opset_version=OPSET,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        onnx.checker.check_model(onnx.load(path))
        ort_session(path).run(None, {"input": dummy.numpy()})
        print(f"[ONNX] QAT model exported with QuantizeLinear/DequantizeLinear nodes -> {path}")
        return "fake-quant -> QDQ export"
    except Exception as e:
        print(f"[ONNX] Direct QAT export not supported ({type(e).__name__}: {str(e)[:100]})")
        print("[ONNX] Fallback: export QAT-trained weights as float graph + ORT static INT8 quantization")
        m.apply(disable_fake_quant)
        tmp = path.replace(".onnx", "_float_tmp.onnx")
        export_float(m, tmp)
        ort_static_quantize(tmp, path)
        os.remove(tmp)
        print(f"[ONNX] QAT INT8 model -> {path}")
        return "QAT weights -> ORT static quantization"


def main():
    parser = argparse.ArgumentParser(description="Export all models to ONNX")
    parser.add_argument("--calib-images", type=int, default=1000)
    args = parser.parse_args()

    start_logging("export_onnx")
    set_seed(42)
    xs, _ = load_test_tensors()
    xs = xs[:256]

    for name in ("fp32", "unstructured", "structured"):
        if not os.path.exists(CKPT[name]):
            print(f"[skip] {name}: checkpoint not found")
            continue
        model = load_model(name, "cpu")
        export_float(model, ONNX_PATHS[name])
        diff = max_abs_diff(model, ONNX_PATHS[name], xs)
        print(f"[ONNX] {name:13s} -> {os.path.basename(ONNX_PATHS[name]):36s} "
              f"{mb(os.path.getsize(ONNX_PATHS[name])):7.2f} MB | max |torch - onnx| = {diff:.2e}")

    if os.path.exists(ONNX_PATHS["fp32"]):
        print("\n[ONNX] PTQ INT8: ONNX Runtime static quantization (QDQ, per-channel) ...")
        ort_static_quantize(ONNX_PATHS["fp32"], ONNX_PATHS["ptq"], args.calib_images)
        agree = top1_agreement(ONNX_PATHS["fp32"], ONNX_PATHS["ptq"], xs)
        print(f"[ONNX] ptq           -> {os.path.basename(ONNX_PATHS['ptq']):36s} "
              f"{mb(os.path.getsize(ONNX_PATHS['ptq'])):7.2f} MB | top-1 agreement with FP32 = {agree:.1f}%")

    if os.path.exists(ONNX_PATHS["qat"]):
        print(f"[ONNX] qat           -> {os.path.basename(ONNX_PATHS['qat']):36s} "
              f"{mb(os.path.getsize(ONNX_PATHS['qat'])):7.2f} MB (created by qat.py)")
    else:
        print("[ONNX] qat: not found - run  python src\\qat.py  first")


if __name__ == "__main__":
    main()
