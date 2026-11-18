"""
calibration.py
--------------
PTQ calibration pipeline: a fixed, reproducible subset of the CIFAR-10 TRAINING
set (no augmentation) is streamed through a model with observers inserted, so
every observer records the activation range used to compute scale/zero-point.

Standalone use (prints the recorded activation ranges):
    python src\calibration.py --images 2000
"""
import argparse

import torch
from torch.ao.quantization import get_default_qconfig, prepare
from torch.utils.data import DataLoader, Subset

from common import get_train_set, load_model, set_quant_engine, start_logging


def get_calibration_loader(num_images=2000, batch_size=100, seed=42):
    dataset = get_train_set(augment=False)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:num_images].tolist()
    return DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def calibrate(model, loader):
    model.eval()
    seen = 0
    for i, (x, _) in enumerate(loader, 1):
        model(x)
        seen += x.size(0)
        if i % 5 == 0 or i == len(loader):
            print(f"    calibration batch {i:3d}/{len(loader)}  ({seen} images)")
    return seen


def main():
    parser = argparse.ArgumentParser(description="Run calibration and show observer ranges")
    parser.add_argument("--images", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()

    start_logging("calibration")
    engine = set_quant_engine()
    model = load_model("fp32", "cpu")
    model.fuse_model(is_qat=False)
    model.qconfig = get_default_qconfig(engine)
    prepare(model, inplace=True)
    calibrate(model, get_calibration_loader(args.images, args.batch))

    print(f"\n{'observer':55s} {'min':>10s} {'max':>10s}")
    for name, mod in model.named_modules():
        if name.endswith("activation_post_process") and hasattr(mod, "min_val") and mod.min_val.numel() == 1:
            print(f"{name:55s} {mod.min_val.item():10.4f} {mod.max_val.item():10.4f}")


if __name__ == "__main__":
    main()
