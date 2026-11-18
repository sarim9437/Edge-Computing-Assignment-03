# Assignment 3 - Edge Computing & Intelligent Systems

Model optimization of ResNet-18 on CIFAR-10 with PyTorch:

* **Question 1** - Post-Training Static Quantization (PTQ) and Quantization-Aware Training (QAT), FP32 -> INT8
* **Question 2** - Unstructured iterative magnitude pruning and structured channel pruning with architecture reconstruction
* **Deployment** - TorchScript checkpoints, ONNX export, CPU inference with ONNX Runtime, automated benchmarking (CSV)

## 1. Requirements

* Windows 10/11 (x86-64), **Python 3.11** (3.10-3.12 also work; 3.13 is not supported by torch 2.5.1)
* CPU only is enough (all results were produced on CPU). A CUDA GPU is used automatically if available.
* Packages (requirements.txt): torch 2.5.1, torchvision 0.20.1, onnx 1.17.0, onnxruntime 1.20.1, numpy, pillow, matplotlib

## 2. Installation (PowerShell, inside the project folder)

    py -3.11 -m venv venv
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    .\venv\Scripts\Activate.ps1
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

CIFAR-10 is downloaded automatically into data\ on the first run.

## 3. Run the whole pipeline (reproducible)

    powershell -ExecutionPolicy Bypass -File .\run_all.ps1          (full run, about 3 h on CPU)
    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Quick   (quick smoke test)

All random seeds are fixed (42) and the calibration subset is fixed, so runs are repeatable.

## 4. Step-by-step execution

| # | Command | What it does | Main outputs |
|---|---------|--------------|--------------|
| 1 | python src\train_baseline.py --epochs 15 | Train FP32 ResNet-18 | models\resnet18_fp32.pth |
| 2 | python src\ptq.py | Fusion, fbgemm backend, calibration (2000 train images), INT8 convert | models\resnet18_ptq_int8_scripted.pt |
| 3 | python src\qat.py | Fake-quant insertion, QAT fine-tuning, INT8 convert, ONNX export | models\resnet18_qat_int8_scripted.pt |
| 4 | python src\prune_unstructured.py | Global L1 magnitude pruning, 4 iterations to 80% sparsity, fine-tuning, permanent | models\resnet18_unstructured_pruned.pth |
| 5 | python src\prune_structured.py | L1 channel pruning (50%), rebuilt smaller architecture, fine-tuning | models\resnet18_structured_pruned.pth |
| 6 | python src\export_onnx.py | ONNX export + ORT static INT8 quantization | onnx\*.onnx |
| 7 | python src\benchmark.py --suite all | Accuracy, size, sparsity, MACs, latency, throughput | results\*_benchmark_summary.csv, results\*_test_predictions.csv |
| 8 | python src\onnx_infer.py | ONNX Runtime CPU inference + benchmark | results\onnx_benchmark.csv |
| 9 | python src\infer.py --model all --num 10 | Inference pipeline for all five models | console + logs\infer.log |

Single image:

    python src\infer.py --model ptq --image samples\test_000_cat.png
    python src\onnx_infer.py --image samples\test_000_cat.png

Optional standalone calibration report: python src\calibration.py

## 5. Project structure

    src\common.py              model (quantization-ready ResNet-18), data, helpers, model loader
    src\train_baseline.py      FP32 baseline training
    src\calibration.py         PTQ calibration pipeline
    src\ptq.py                 post-training static quantization
    src\finetune.py            shared fine-tuning pipeline (QAT + pruning)
    src\qat.py                 quantization-aware training
    src\prune_unstructured.py  iterative unstructured magnitude pruning
    src\prune_structured.py    structured channel pruning + architecture reconstruction
    src\export_onnx.py         ONNX export
    src\infer.py               PyTorch inference pipeline (all five models)
    src\onnx_infer.py          ONNX Runtime CPU inference + benchmark
    src\benchmark.py           automated benchmarking -> CSV
    models\                    5 executable checkpoints (+ INT8 state_dicts)
    onnx\                      5 ONNX models
    results\                   CSV results, metrics JSON, comparison chart
    logs\                      console log of every step
    run_all.ps1                reproducible end-to-end pipeline

## 6. Implementation notes

* **Model**: CIFAR ResNet-18 (11.17 M parameters). The stem convolution uses stride 2 to make CPU training 4x cheaper (138.9 M MACs per image). QuantStub/DeQuantStub and FloatFunctional (residual add+ReLU) make it quantizable in eager mode.
* **PTQ**: Conv+BN+ReLU fusion, fbgemm backend, HistogramObserver (activations) + per-channel symmetric weights, calibration on 2000 fixed training images.
* **QAT**: initialised from FP32, fuse_modules_qat, FakeQuantize on weights and activations, 2 epochs fine-tuning (lr 1e-3), observers and BN statistics frozen in the last epoch.
* **Unstructured pruning**: global L1 ranking over all Conv/Linear weights, 4 iterations x 33.1% of the remaining weights = 80% sparsity, masks during fine-tuning, prune.remove makes it permanent. Zeros are stored densely, so the saving shows in the compressed (gzip) size.
* **Structured pruning**: the middle channels of every residual block (conv1 output / conv2 input) are ranked by filter L1 norm and 50% are removed. Block output channels are kept because residual additions tie them together, so all tensor shapes stay valid. A new, smaller ResNet18 is built, so no masked zero channels are used at inference.
* **ONNX**: FP32 / pruned models exported directly (opset 17). PyTorch eager INT8 kernels are not standard ONNX ops, so INT8 ONNX models use ONNX Runtime static quantization (QDQ, per-channel) with the same calibration subset (QAT: QAT-trained weights).

## 7. Results (CPU, CIFAR-10 test set)

| Model | Top-1 accuracy | Size |
|-------|----------------|------|
| FP32 baseline | 89.77% | 42.70 MB |
| PTQ INT8 | 89.87% | 10.78 MB (3.96x smaller) |
| QAT INT8 | 89.70% | 10.78 MB (3.96x smaller) |
| Unstructured pruned (80% sparse) | 89.98% | 42.70 MB dense / 11.64 MB gzip |
| Structured pruned (50% channels) | 88.75% | 21.73 MB, 49.3% fewer MACs |

Latency and throughput: see results\all_models_summary.csv and results\onnx_benchmark.csv.
