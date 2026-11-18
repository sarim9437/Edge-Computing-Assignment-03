<#
 run_all.ps1 - Reproducible end-to-end pipeline for Assignment 3
 train -> PTQ -> QAT -> unstructured pruning -> structured pruning -> ONNX -> benchmarks -> inference

 Usage (from the project folder):
   powershell -ExecutionPolicy Bypass -File .\run_all.ps1           # full run (CPU: about 3 hours)
   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Quick    # quick smoke test
#>
param(
    [int]$Epochs = 15,
    [int]$QatEpochs = 2,
    [int]$PruneIterations = 4,
    [int]$StructFtEpochs = 3,
    [switch]$Quick
)

Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$iterFt = 1
$finalFt = 1
if ($Quick) {
    $Epochs = 1; $QatEpochs = 1; $PruneIterations = 2; $StructFtEpochs = 1; $iterFt = 0; $finalFt = 1
}

function Step([string]$Title, [string[]]$ArgList) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    & $py @ArgList
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at: $Title" -ForegroundColor Red
        exit 1
    }
}

$start = Get-Date
Step "1/9 Train FP32 baseline" @("src\train_baseline.py", "--epochs", "$Epochs")
Step "2/9 Post-training static quantization (PTQ)" @("src\ptq.py")
Step "3/9 Quantization-aware training (QAT)" @("src\qat.py", "--epochs", "$QatEpochs")
Step "4/9 Unstructured iterative pruning" @("src\prune_unstructured.py", "--iterations", "$PruneIterations", "--ft-epochs", "$iterFt", "--final-ft-epochs", "$finalFt")
Step "5/9 Structured channel pruning" @("src\prune_structured.py", "--ft-epochs", "$StructFtEpochs")
Step "6/9 ONNX export" @("src\export_onnx.py")
Step "7/9 PyTorch benchmark (Q1 + Q2)" @("src\benchmark.py", "--suite", "all")
Step "8/9 ONNX Runtime CPU inference + benchmark" @("src\onnx_infer.py")
Step "9/9 Inference pipeline demo" @("src\infer.py", "--model", "all", "--num", "10")
$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host ("All steps finished in {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Green
