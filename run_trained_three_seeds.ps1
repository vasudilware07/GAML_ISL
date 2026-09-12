$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw "Virtual environment not found at .\.venv. Create it and install requirements first."
}

if (-not (Test-Path ".\results")) {
    New-Item -ItemType Directory -Path ".\results" | Out-Null
}

& ".\.venv\Scripts\Activate.ps1"

python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

Start-Transcript -Path ".\results\within_domain_trained_full_gpu_3seeds_600ep.log" -Append

try {
    python tools\run_within_domain_trained.py `
    --datasets asl_alphabet arabic_sign_alphabet libras_alphabet thai_fingerspelling `
    --encoders mlp transformer `
    --representations raw angle raw_angle `
    --shots 1 3 5 `
    --episodes_train 600 `
    --episodes_eval 600 `
    --epochs 5 `
    --q_query 5 `
    --seeds 42 1337 2024 `
    --device cuda `
    --auto_adjust_q `
    --output results\within_domain_trained_full_gpu_3seeds_600ep.csv `
    --resume
}
finally {
    Stop-Transcript
}
