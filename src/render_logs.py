"""
render_logs.py
--------------
Renders the console logs in logs/ (captured verbatim by common.start_logging)
as terminal-style PNG images in report/screenshots/, so the report can show the
output of every step without manual screen captures.

Run:  python src\render_logs.py
"""
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

from common import BASE_DIR, LOG_DIR

SHOT_DIR = os.path.join(BASE_DIR, "report", "screenshots")
PROMPT = f"(venv) PS {BASE_DIR}> "

# (output image name, log file, command that produced it, optional line to start from)
JOBS = [
    ("01_baseline_training", "train_baseline.log", "python src\\train_baseline.py --epochs 15", None),
    ("02_ptq", "ptq.log", "python src\\ptq.py", None),
    ("03_qat", "qat.log", "python src\\qat.py", "[2] QAT fine-tuning"),
    ("04_unstructured_pruning", "prune_unstructured.log", "python src\\prune_unstructured.py", None),
    ("05_structured_pruning", "prune_structured.log", "python src\\prune_structured.py", None),
    ("06_onnx_export", "export_onnx.log", "python src\\export_onnx.py", None),
    ("07_benchmark", "benchmark_all.log", "python src\\benchmark.py --suite all", None),
    ("08_onnx_cpu_inference", "onnx_infer.log", "python src\\onnx_infer.py", None),
    ("09_inference_pipeline", "infer.log", "python src\\infer.py --model all --num 10", None),
]

MAX_LINES = 70      # longer logs keep the header and the last lines
WRAP = 180          # wrap very long lines
FONT_SIZE = 16
PAD = 18
BG = (12, 12, 12)
FG = (204, 204, 204)
PROMPT_COLOR = (230, 230, 120)
HEADER_COLOR = (97, 214, 214)


def load_font(size):
    for path in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def prepare_lines(log_path, command, start_marker):
    with open(log_path, encoding="utf-8") as fh:
        raw = fh.read().splitlines()
    if start_marker:
        idx = next((i for i, l in enumerate(raw) if l.strip().startswith(start_marker)), None)
        if idx is not None:
            raw = raw[:4] + ["..."] + raw[idx:]
    if len(raw) > MAX_LINES:
        raw = raw[:4] + ["..."] + raw[-(MAX_LINES - 5):]
    lines = [PROMPT + command]
    for line in raw:
        lines.extend(textwrap.wrap(line, WRAP, drop_whitespace=False) or [""])
    return lines


def render(lines, out_path):
    font = load_font(FONT_SIZE)
    char_w = font.getlength("M")
    line_h = int(FONT_SIZE * 1.4)
    width = int(max(len(l) for l in lines) * char_w) + 2 * PAD
    height = len(lines) * line_h + 2 * PAD
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        if line.startswith(PROMPT):
            color = PROMPT_COLOR
        elif line.startswith("====") or line.startswith("["):
            color = HEADER_COLOR
        else:
            color = FG
        draw.text((PAD, PAD + i * line_h), line, font=font, fill=color)
    img.save(out_path)


def main():
    os.makedirs(SHOT_DIR, exist_ok=True)
    for name, log, command, marker in JOBS:
        log_path = os.path.join(LOG_DIR, log)
        if not os.path.exists(log_path):
            print(f"[skip] {log} not found")
            continue
        out = os.path.join(SHOT_DIR, f"{name}.png")
        render(prepare_lines(log_path, command, marker), out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
