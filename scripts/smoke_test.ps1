# Smoke test: starts the API, fires real predictions, checks stats.
# Run: .\scripts\smoke_test.ps1

$ErrorActionPreference = "Stop"
$maxWaitSeconds = 60

# Pre-flight: kill anything squatting on port 8000 from previous failed runs
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

Write-Host "[1/5] Starting API..." -ForegroundColor Cyan
$apiProcess = Start-Process -FilePath "uv" `
    -ArgumentList "run", "uvicorn", "loan_mlops.api.app:app", "--port", "8000" `
    -WorkingDirectory $PWD `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput "smoke-stdout.log" `
    -RedirectStandardError "smoke-stderr.log"

$ready = $false
for ($i = 0; $i -lt ($maxWaitSeconds * 2); $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $h = Invoke-RestMethod -Uri http://127.0.0.1:8000/health -TimeoutSec 1
        if ($h.status -eq "ok") {
            $ready = $true
            Write-Host "    Ready after $([math]::Round($i * 0.5, 1))s" -ForegroundColor DarkGray
            break
        }
    } catch { }
}

if (-not $ready) {
    Write-Host "API never became ready. Last stderr:" -ForegroundColor Red
    if (Test-Path "smoke-stderr.log") { Get-Content "smoke-stderr.log" -Tail 30 }
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    throw "API did not become healthy within $maxWaitSeconds seconds"
}

try {
    Write-Host "[2/5] Health check..." -ForegroundColor Cyan
    $health = Invoke-RestMethod -Uri http://127.0.0.1:8000/health
    Write-Host "    model_loaded=$($health.model_loaded), version=$($health.model_version)" -ForegroundColor Green

    Write-Host "[3/5] Firing 30 predictions..." -ForegroundColor Cyan
    $approved = 0; $declined = 0
    1..30 | ForEach-Object {
        $body = @{
            application_id = "smoke-$_"
            CODE_GENDER = "M"
            DAYS_BIRTH = -12000
            CNT_CHILDREN = 1
            NAME_INCOME_TYPE = "Working"
            NAME_EDUCATION_TYPE = "Higher education"
            AMT_INCOME_TOTAL = 180000
            DAYS_EMPLOYED = -2000
            OCCUPATION_TYPE = "Core staff"
            NAME_CONTRACT_TYPE = "Cash loans"
            AMT_CREDIT = 500000
            AMT_ANNUITY = 24000
            AMT_GOODS_PRICE = 450000
            EXT_SOURCE_1 = 0.7
            EXT_SOURCE_2 = 0.6
            EXT_SOURCE_3 = 0.5
        } | ConvertTo-Json
        $r = Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType "application/json"
        if ($r.decision -eq "approve") { $approved++ } else { $declined++ }
    }
    Write-Host "    Approved: $approved, Declined: $declined" -ForegroundColor Green

    Write-Host "[4/5] Reading /stats..." -ForegroundColor Cyan
    $stats = Invoke-RestMethod -Uri http://127.0.0.1:8000/stats
    $stats.by_cohort | ForEach-Object {
        Write-Host ("    {0} [{1}]: n={2}, avg_proba={3}, decline_rate={4}, latency_ms={5}" -f `
            $_.cohort, $_.model_version, $_.predictions, $_.avg_default_probability, $_.decline_rate, $_.avg_latency_ms) -ForegroundColor Green
    }

    Write-Host "[5/5] Verifying correlation IDs propagate..." -ForegroundColor Cyan
    $cid = "smoke-trace-$(Get-Random)"
    $body = @{
        application_id = "smoke-corr-check"
        CODE_GENDER = "F"; DAYS_BIRTH = -10000; CNT_CHILDREN = 0
        NAME_INCOME_TYPE = "Working"; NAME_EDUCATION_TYPE = "Secondary / secondary special"
        AMT_INCOME_TOTAL = 90000; DAYS_EMPLOYED = -1500; OCCUPATION_TYPE = "Laborers"
        NAME_CONTRACT_TYPE = "Cash loans"; AMT_CREDIT = 200000; AMT_ANNUITY = 12000
        AMT_GOODS_PRICE = 180000; EXT_SOURCE_1 = 0.3; EXT_SOURCE_2 = 0.4; EXT_SOURCE_3 = 0.5
    } | ConvertTo-Json
    $resp = Invoke-WebRequest -Uri http://127.0.0.1:8000/predict -Method Post -Body $body `
        -ContentType "application/json" -Headers @{ "x-correlation-id" = $cid }
    $returned = $resp.Headers["x-correlation-id"]
    if ($returned -ne $cid) {
        throw "Correlation ID round-trip failed: sent $cid, got $returned"
    }
    Write-Host "    Correlation ID round-tripped correctly" -ForegroundColor Green

    Write-Host "`nSmoke test PASSED" -ForegroundColor Green
}
finally {
    Write-Host "`nStopping API..." -ForegroundColor Cyan
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -Path "smoke-stdout.log", "smoke-stderr.log" -ErrorAction SilentlyContinue
}