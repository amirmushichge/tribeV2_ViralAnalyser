param(
    [switch]$NoBrowser
)

$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $appDir ".venv\Scripts\python.exe"
$requirementsFile = Join-Path $appDir "requirements.txt"
$bootstrapScript = Join-Path $appDir "bootstrap_models.py"
$bootstrapReadyFile = Join-Path $appDir ".bootstrap\models-ready.json"
$hostAddress = "127.0.0.1"
$preferredPort = 8000
$fallbackPorts = 8001..8010
$cudaTorchIndexUrl = "https://download.pytorch.org/whl/cu126"
$torchVersion = "2.6.0"
$torchvisionVersion = "0.21.0"
$torchaudioVersion = "2.6.0"
$minCudaInstallFreeGb = 8

function Stop-WithMessage {
    param(
        [string]$Message
    )

    Write-Host ""
    Write-Host $Message -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}

function New-LocalVenv {
    $siblingPython = Join-Path (Split-Path -Parent $appDir) "tribe311_clean\Scripts\python.exe"
    if (Test-Path $siblingPython) {
        & $siblingPython -m venv ".venv"
        return $LASTEXITCODE -eq 0
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.11 -m venv ".venv"
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        & python -m venv ".venv"
        return $LASTEXITCODE -eq 0
    }

    return $false
}

function Test-AppUrl {
    param(
        [int]$Port
    )

    try {
        $response = Invoke-WebRequest -Uri ("http://{0}:{1}" -f $hostAddress, $Port) -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Test-TribeAppUrl {
    param(
        [int]$Port
    )

    try {
        $response = Invoke-WebRequest -Uri ("http://{0}:{1}" -f $hostAddress, $Port) -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 500) {
            return $false
        }
        $content = [string]$response.Content
        return $content -like "*TRIBE Review MVP*" -or $content -like "*Predict virality with Meta TRIBE v2*"
    } catch {
        return $false
    }
}

function Test-PortFree {
    param(
        [int]$Port
    )

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($hostAddress), $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Get-LaunchPort {
    if (Test-TribeAppUrl -Port $preferredPort) {
        return $preferredPort
    }

    if (Test-PortFree -Port $preferredPort) {
        return $preferredPort
    }

    foreach ($port in $fallbackPorts) {
        if (Test-PortFree -Port $port) {
            return $port
        }
    }

    return $null
}

function Test-NvidiaGpu {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        & nvidia-smi --query-gpu=name --format=csv,noheader 1>$null 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }

    try {
        $controllers = Get-CimInstance Win32_VideoController -ErrorAction Stop
        foreach ($controller in $controllers) {
            if ([string]$controller.Name -like "*NVIDIA*") {
                return $true
            }
        }
    } catch {
    }

    return $false
}

function Test-TorchCuda {
    if (-not (Test-Path $python)) {
        return $false
    }

    & $python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 1>$null 2>$null
    return $LASTEXITCODE -eq 0
}

function Test-PythonDependencies {
    if (-not (Test-Path $python)) {
        return $false
    }

    & $python -c "import re, sys; import fastapi, uvicorn, torch, playwright; import PIL.Image; m = re.match(r'(\d+)\.(\d+)\.(\d+)', torch.__version__); v = tuple(map(int, m.groups())) if m else (0, 0, 0); sys.exit(1) if not ((2, 5, 1) <= v < (2, 7, 0)) else None; import tribev2" 1>$null 2>$null
    return $LASTEXITCODE -eq 0
}

function Get-AppDriveFreeGb {
    try {
        $root = [System.IO.Path]::GetPathRoot($appDir)
        $drive = Get-PSDrive -Name $root.Substring(0, 1) -ErrorAction Stop
        return [math]::Round(($drive.Free / 1GB), 1)
    } catch {
        return $null
    }
}

function Install-CudaTorch {
    Write-Host ""
    Write-Host "NVIDIA GPU detected, but PyTorch CUDA is not active." -ForegroundColor Yellow
    Write-Host "Installing CUDA-enabled PyTorch. This can take several minutes and only happens when needed." -ForegroundColor Cyan

    $freeGb = Get-AppDriveFreeGb
    if ($null -ne $freeGb -and $freeGb -lt $minCudaInstallFreeGb) {
        Stop-WithMessage ("Not enough free disk space to install CUDA-enabled PyTorch. Free at least {0} GB on drive {1}, then run Start_TRIBE_Review.cmd again. Current free space: {2} GB." -f $minCudaInstallFreeGb, ([System.IO.Path]::GetPathRoot($appDir)), $freeGb)
    }

    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "Could not upgrade pip before installing CUDA-enabled PyTorch."
    }

    & $python -m pip install --upgrade --force-reinstall "torch==$torchVersion" "torchvision==$torchvisionVersion" "torchaudio==$torchaudioVersion" --index-url $cudaTorchIndexUrl
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "Could not install CUDA-enabled PyTorch. If the terminal shows 'No space left on device', free disk space and run Start_TRIBE_Review.cmd again. Otherwise update the NVIDIA driver and try again."
    }

    if (-not (Test-TorchCuda)) {
        Stop-WithMessage "NVIDIA GPU was detected, but PyTorch still cannot use CUDA. Update the NVIDIA driver, then run Start_TRIBE_Review.cmd again."
    }
}

Set-Location $appDir

if (-not (Test-Path $venvPython)) {
    Write-Host ""
    Write-Host "Creating local Python environment: .venv" -ForegroundColor Cyan
    $venvCreated = New-LocalVenv
    if (-not $venvCreated -or -not (Test-Path $venvPython)) {
        Stop-WithMessage "Could not create .venv. Install Python 3.11, then run Start_TRIBE_Review.cmd again."
    }
}

$python = $venvPython

if (Test-NvidiaGpu) {
    if (-not (Test-TorchCuda)) {
        Install-CudaTorch
    }
}

if (-not (Test-PythonDependencies)) {
    if (-not (Test-Path $requirementsFile)) {
        Stop-WithMessage "requirements.txt not found. Download the full repository archive again."
    }

    Write-Host ""
    Write-Host "Installing Python dependencies. First run can take several minutes." -ForegroundColor Cyan
    Write-Host "Please keep this terminal open. This only happens during the initial setup." -ForegroundColor Cyan
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "Could not upgrade pip inside .venv."
    }

    & $python -m pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "Could not install dependencies from requirements.txt."
    }
}

if (Test-NvidiaGpu) {
    if (-not (Test-TorchCuda)) {
        Install-CudaTorch
    }
}

if (-not (Test-Path $bootstrapReadyFile)) {
    if (-not (Test-Path $bootstrapScript)) {
        Stop-WithMessage "bootstrap_models.py not found. Download the full repository archive again."
    }

    & $python $bootstrapScript
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "Initial model setup failed. Fix the issue above, then run Start_TRIBE_Review.cmd again."
    }

    Write-Host ""
    Write-Host "Initial setup finished successfully." -ForegroundColor Green
    Write-Host "Close this window and run Start_TRIBE_Review.cmd one more time to start the app." -ForegroundColor Green
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 0
}

$port = Get-LaunchPort
if (-not $port) {
    Stop-WithMessage "Could not find a free port from 8000 to 8010."
}

$url = "http://{0}:{1}" -f $hostAddress, $port

if (Test-TribeAppUrl -Port $port) {
    Write-Host ""
    Write-Host ("TRIBE Review already running on {0}" -f $url) -ForegroundColor Green
    if (-not $NoBrowser) {
        Start-Process $url
    }
    exit 0
}

Write-Host ""
Write-Host ("Starting TRIBE Review on {0}" -f $url) -ForegroundColor Cyan

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($LaunchUrl)
        $deadline = (Get-Date).AddMinutes(3)
        while ((Get-Date) -lt $deadline) {
            try {
                $response = Invoke-WebRequest -Uri $LaunchUrl -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                    Start-Process $LaunchUrl
                    return
                }
            } catch {
            }
            Start-Sleep -Milliseconds 750
        }
    } -ArgumentList $url | Out-Null
}

& $python -m uvicorn app:app --host $hostAddress --port $port
