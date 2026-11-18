"""
common.py
---------
Shared building blocks for Assignment 3 (Edge Computing & Intelligent Systems).

  * ResNet-18 adapted to CIFAR-10 (32x32 input) and made quantization-ready
    (QuantStub / DeQuantStub, FloatFunctional for the residual add, fuse_model()).
  * The same class accepts a custom number of "middle" channels per residual
    block. Structured pruning uses this to rebuild a genuinely thinner network.
  * CIFAR-10 data pipeline, train / evaluate helpers, size / sparsity / MAC
    counters and a single load_model() for all five model variants.
"""
import gzip
import io
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat
from torch.utils.data import DataLoader

# Keep console output clean (quantization / ONNX code emits many advisory warnings)
for _cat in (UserWarning, FutureWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=_cat)

# ------------------------------------------------------------------ paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
ONNX_DIR = os.path.join(BASE_DIR, "onnx")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAMPLE_DIR = os.path.join(BASE_DIR, "samples")
for _d in (DATA_DIR, MODEL_DIR, RESULT_DIR, ONNX_DIR, LOG_DIR, SAMPLE_DIR):
    os.makedirs(_d, exist_ok=True)

MODEL_NAMES = ["fp32", "ptq", "qat", "unstructured", "structured"]

# Executable checkpoints (single source of truth for every script)
CKPT = {
    "fp32": os.path.join(MODEL_DIR, "resnet18_fp32.pth"),
    "ptq": os.path.join(MODEL_DIR, "resnet18_ptq_int8_scripted.pt"),
    "qat": os.path.join(MODEL_DIR, "resnet18_qat_int8_scripted.pt"),
    "unstructured": os.path.join(MODEL_DIR, "resnet18_unstructured_pruned.pth"),
    "structured": os.path.join(MODEL_DIR, "resnet18_structured_pruned.pth"),
}
INT8_STATE = {
    "ptq": os.path.join(MODEL_DIR, "resnet18_ptq_int8_state_dict.pth"),
    "qat": os.path.join(MODEL_DIR, "resnet18_qat_int8_state_dict.pth"),
}
ONNX_PATHS = {
    "fp32": os.path.join(ONNX_DIR, "resnet18_fp32.onnx"),
    "ptq": os.path.join(ONNX_DIR, "resnet18_ptq_int8.onnx"),
    "qat": os.path.join(ONNX_DIR, "resnet18_qat_int8.onnx"),
    "unstructured": os.path.join(ONNX_DIR, "resnet18_unstructured_pruned.onnx"),
    "structured": os.path.join(ONNX_DIR, "resnet18_structured_pruned.onnx"),
}
DESCRIPTIONS = {
    "fp32": "Baseline ResNet-18 (FP32)",
    "ptq": "Post-training static quantization (INT8)",
    "qat": "Quantization-aware training (INT8)",
    "unstructured": "Iterative global magnitude pruning (FP32, sparse)",
    "structured": "L1 channel pruning, rebuilt architecture (FP32)",
}

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# ------------------------------------------------------------------ logging
class _Tee:
    """Duplicate everything printed to the console into a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def start_logging(name):
    log_file = open(os.path.join(LOG_DIR, f"{name}.log"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    dev = ("cuda: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {name} | torch {torch.__version__} | {dev}")
    print("=" * 78)


# ------------------------------------------------------------------ reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ data
def train_transform():
    return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def test_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def get_train_set(augment=True):
    tf = train_transform() if augment else test_transform()
    return torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf)


def get_test_set():
    return torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform())


def get_loaders(batch_size=128, workers=2):
    pin = torch.cuda.is_available()
    train_loader = DataLoader(get_train_set(True), batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    test_loader = DataLoader(get_test_set(), batch_size=256, shuffle=False,
                             num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    return train_loader, test_loader


def get_test_loader(batch_size=256):
    return DataLoader(get_test_set(), batch_size=batch_size, shuffle=False, num_workers=0)


def load_test_tensors():
    """Whole CIFAR-10 test set as two tensors (so benchmarks time only the model)."""
    xs, ys = [], []
    for x, y in DataLoader(get_test_set(), batch_size=1000, shuffle=False):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    return test_transform()(img).unsqueeze(0)


# ------------------------------------------------------------------ model
STAGE_WIDTHS = [64, 64, 128, 128, 256, 256, 512, 512]
STAGE_STRIDES = [1, 1, 2, 1, 2, 1, 2, 1]


class BasicBlock(nn.Module):
    """ResNet basic block. mid_c = channels between conv1 and conv2 (the prunable width)."""

    def __init__(self, in_c, mid_c, out_c, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                                            nn.BatchNorm2d(out_c))
        # FloatFunctional makes the residual "add + relu" quantizable
        self.skip_add = nnq.FloatFunctional()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.skip_add.add_relu(out, identity)


class ResNet18(nn.Module):
    """CIFAR-10 ResNet-18 (3x3 stem, no max-pool), quantization-ready."""

    def __init__(self, num_classes=10, mid_channels=None):
        super().__init__()
        self.mid_channels = list(mid_channels) if mid_channels else list(STAGE_WIDTHS)
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)  # stride 2 -> 4x fewer FLOPs (CPU-friendly)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        blocks, in_c = [], 64
        for mid, out_c, stride in zip(self.mid_channels, STAGE_WIDTHS, STAGE_STRIDES):
            blocks.append(BasicBlock(in_c, mid, out_c, stride))
            in_c = out_c
        self.layers = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    def fuse_model(self, is_qat=False):
        """Fuse Conv+BN+ReLU and Conv+BN (eval mode for PTQ, train mode for QAT)."""
        fuse = fuse_modules_qat if is_qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for block in self.layers:
            fuse(block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            if block.downsample is not None:
                fuse(block.downsample, [["0", "1"]], inplace=True)


def set_quant_engine():
    engines = torch.backends.quantized.supported_engines
    engine = "fbgemm" if "fbgemm" in engines else "qnnpack"
    torch.backends.quantized.engine = engine
    return engine


# ------------------------------------------------------------------ train / eval
def train_one_epoch(model, loader, optimizer, device, scheduler=None, use_amp=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        out = model(x.to(device))
        correct += (out.argmax(1).cpu() == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def predict(model, xs, batch_size=256):
    outs = []
    for i in range(0, len(xs), batch_size):
        outs.append(model(xs[i:i + batch_size]).float())
    return torch.cat(outs)


# ------------------------------------------------------------------ size / sparsity / MACs
def state_dict_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def checkpoint_bytes(name, model):
    """Serialized weights. INT8 models use their saved INT8 state_dict file."""
    if name in INT8_STATE:
        with open(INT8_STATE[name], "rb") as f:
            return f.read()
    return state_dict_bytes(model)


def mb(nbytes):
    return nbytes / (1024 ** 2)


def gzip_mb(data):
    return mb(len(gzip.compress(data, compresslevel=6)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def weight_stats(model):
    """(total, non-zero) weights of all Conv2d / Linear layers."""
    total = nonzero = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach()
            total += w.numel()
            nonzero += int((w != 0).sum())
    return total, nonzero


@torch.no_grad()
def count_macs(model):
    """Dense multiply-accumulate operations for one 32x32 image."""
    macs = [0]

    def conv_hook(m, inp, out):
        macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    model(torch.randn(1, 3, 32, 32, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    return macs[0]


# ------------------------------------------------------------------ save / load
def save_float_checkpoint(model, name, **meta):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "mid_channels": list(model.mid_channels), **meta}, CKPT[name])


def load_model(name, device="cpu"):
    """Load any of the five models ready for inference."""
    if not os.path.exists(CKPT[name]):
        raise FileNotFoundError(f"{CKPT[name]} not found - run the script that creates '{name}' first.")
    if name in ("ptq", "qat"):
        set_quant_engine()
        model = torch.jit.load(CKPT[name], map_location="cpu")  # INT8 runs on CPU
        model.eval()
        return model
    ckpt = torch.load(CKPT[name], map_location="cpu", weights_only=True)
    model = ResNet18(mid_channels=ckpt.get("mid_channels"))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# ------------------------------------------------------------------ pretty printing
def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_table(rows, columns):
    widths = [max([len(c)] + [len(_fmt(r.get(c, ""))) for r in rows]) for c in columns]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    print("| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |")
    print(line)
    for r in rows:
        print("| " + " | ".join(_fmt(r.get(c, "")).rjust(w) for c, w in zip(columns, widths)) + " |")
    print(line)
