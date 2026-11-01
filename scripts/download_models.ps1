param(
    [string]$ModelsDir = (Join-Path $PSScriptRoot "..\models"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ModelsDir = [System.IO.Path]::GetFullPath($ModelsDir)

Write-Host "=================================================="
Write-Host "      MangaTranslate Model Downloader            "
Write-Host "=================================================="
Write-Host "Target Directory: $ModelsDir"

# 1. Comic Speech Bubble Detector
$detDir = Join-Path $ModelsDir "detector"
New-Item -ItemType Directory -Force -Path $detDir | Out-Null
$detFile = Join-Path $detDir "comic-speech-bubble-detector.pt"
$detUrl = "https://hf-mirror.com/ogkalu/comic-speech-bubble-detector-yolov8m/resolve/main/comic-speech-bubble-detector.pt"
$detExpectedSize = 52079361

if (-not $Force -and (Test-Path $detFile) -and (Get-Item $detFile).Length -eq $detExpectedSize) {
    Write-Host "[1/2] OK: comic-speech-bubble-detector.pt already present ($([math]::Round($detExpectedSize / 1MB, 1)) MB)"
} else {
    Write-Host "[1/2] Downloading comic-speech-bubble-detector.pt..."
    & curl.exe -L --fail --retry 10 --retry-delay 3 -C - -o $detFile $detUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download detector model from $detUrl"
    }
    Write-Host "[1/2] Done detector model!"
}

# 2. Manga-OCR Base Model
$ocrDir = Join-Path $ModelsDir "ocr\manga-ocr-base"
New-Item -ItemType Directory -Force -Path $ocrDir | Out-Null

$ocrSmallFiles = @(
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt"
)

foreach ($f in $ocrSmallFiles) {
    $out = Join-Path $ocrDir $f
    $url = "https://hf-mirror.com/kha-white/manga-ocr-base/resolve/main/$f"
    if ($Force -or -not (Test-Path $out)) {
        Write-Host "Downloading $f..."
        & curl.exe -L --fail --silent --show-error -o $out $url
    }
}

$safeWeights = Join-Path $ocrDir "model.safetensors"
$binWeights = Join-Path $ocrDir "pytorch_model.bin"
$binUrl = "https://hf-mirror.com/kha-white/manga-ocr-base/resolve/main/pytorch_model.bin"
$binExpectedSize = 444135475

if (-not $Force -and (Test-Path $safeWeights) -and (Get-Item $safeWeights).Length -gt 400MB) {
    Write-Host "[2/2] OK: manga-ocr model.safetensors already present ($([math]::Round((Get-Item $safeWeights).Length / 1MB, 1)) MB)"
} else {
    Write-Host "[2/2] Downloading manga-ocr weights (pytorch_model.bin)..."
    & curl.exe -L --fail --retry 15 --retry-delay 3 -C - -o $binWeights $binUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download OCR weights from $binUrl"
    }
    Write-Host "[2/2] Converting PyTorch weights to safetensors format..."
    $py = "python"
    $script = @"
import torch
from pathlib import Path
from safetensors.torch import save_file

bin_p = Path(r'$binWeights')
safe_p = Path(r'$safeWeights')
state_dict = torch.load(bin_p, map_location='cpu', weights_only=True)
for key in ['decoder.cls.predictions.decoder.weight', 'decoder.cls.predictions.decoder.bias']:
    if key in state_dict:
        state_dict[key] = state_dict[key].clone()
save_file(state_dict, safe_p)
print('Successfully saved model.safetensors')
"@
    & $py -c $script
}

Write-Host "=================================================="
Write-Host "All MangaTranslate models are downloaded and ready!"
Write-Host "=================================================="
