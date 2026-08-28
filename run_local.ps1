# Start the RAG API on http://localhost:8000
#
#   .\run_local.ps1
#
# Binds 0.0.0.0 rather than 127.0.0.1 so both http://localhost:8000 and
# http://127.0.0.1:8000 work. Windows resolves "localhost" to the IPv6 address
# ::1 first, which a 127.0.0.1-only bind refuses.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No virtualenv found at .venv. Create one with: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt -e ."
}

Write-Host "RAG API   -> http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Health    -> http://localhost:8000/healthz" -ForegroundColor Green
Write-Host ""

& $python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
