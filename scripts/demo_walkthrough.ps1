# Guided end-to-end demo for the screen recording.
# Run from the repo root:   powershell -ExecutionPolicy Bypass -File scripts\demo_walkthrough.ps1
# Press Enter to move to each next step.

$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$repo = "https://github.com/KirtiTilavat/MLOps-Assignment1"

function Step($n, $title) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host " STEP $n  |  $title" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
}
function Show($cmd) { Write-Host "PS> $cmd" -ForegroundColor Yellow; Invoke-Expression $cmd }
function Next { Read-Host "`n[Enter] to continue" | Out-Null }
function Show-Json($cmd) {
    # Invoke-RestMethod objects of different shapes don't all render in PS 5.1 - print JSON instead.
    Write-Host "PS> $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd | ConvertTo-Json -Depth 5
}
function Wait-Url($url, $seconds = 90) {
    Write-Host "Waiting for $url ..." -ForegroundColor DarkGray
    for ($i = 0; $i -lt $seconds; $i++) {
        try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; Write-Host "ready after ${i}s" -ForegroundColor DarkGray; return }
        catch { Start-Sleep 1 }
    }
    Write-Host "still not ready after ${seconds}s" -ForegroundColor Red
}

# Start the MLflow UI now so it is ready by step 4.
Start-Process -WindowStyle Minimized $py -ArgumentList "-m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"

Step 1 "Project overview - Heart Disease Risk Prediction (MLOps Assignment 1)"
Show "git log --oneline -8"
Show "Get-ChildItem -Name"
Next

Step 2 "Data acquisition and cleaning"
Show "$py scripts/download_data.py"
Show "$py -m src.data_processing"
Next

Step 3 "Model training result (LR vs RF, 5-fold CV, selected by recall)"
Write-Host "PS> Get-Content models\metadata.json | ConvertFrom-Json" -ForegroundColor Yellow
$m = Get-Content models\metadata.json -Raw | ConvertFrom-Json
"Selected model: $($m.model_name)  (by CV $($m.selection_metric))"
foreach ($p in $m.all_models.PSObject.Properties) {
    "{0,-22} CV recall={1:N3}  test accuracy={2:N3}  test ROC-AUC={3:N3}" -f $p.Name, $p.Value.cv.recall, $p.Value.test.accuracy, $p.Value.test.roc_auc
}
Show "$py -m src.predict"
Next

Step 4 "Experiment tracking - MLflow UI (http://127.0.0.1:5000)"
Wait-Url "http://127.0.0.1:5000"
Start-Process "http://127.0.0.1:5000/#/experiments"
Write-Host "Click 'heart-disease-classification' -> expand a parent run (nested candidates) -> select both parents + Compare -> Models tab."
Next

Step 5 "Unit tests and lint"
Show "$py -m pytest -q"
Show "$py -m flake8 src tests scripts app; if (`$?) { 'flake8: OK' }"
Next

Step 6 "CI/CD - GitHub Actions"
Start-Process "$repo/actions"
Write-Host "Show: run history (failure demo #2, then green), open a run: Lint -> Tests -> Train / Docker, artifacts."
Next

Step 7 "Docker - build image and run container with sample input"
Show "docker build -t heart-disease-api:1.2 ."
Show "docker images heart-disease-api"
Show "docker rm -f heart-api-demo 2>`$null; docker run -d -p 8001:8000 --name heart-api-demo heart-disease-api:1.2"
Wait-Url "http://localhost:8001/health"
Show-Json "Invoke-RestMethod http://localhost:8001/health"
Show-Json "Invoke-RestMethod -Method Post -Uri http://localhost:8001/predict -ContentType 'application/json' -InFile sample_request.json"
Show "docker rm -f heart-api-demo"
Next

Step 8 "Kubernetes deployment (Docker Desktop) behind a LoadBalancer"
Show "kubectl config use-context docker-desktop"
Show "kubectl apply -f k8s/namespace.yaml; kubectl apply -f k8s/"
Show "kubectl -n heart-disease rollout status deployment/heart-disease-api"
Show "kubectl -n heart-disease get deploy,pods,svc"
Show-Json "Invoke-RestMethod http://localhost/health"
Show-Json "Invoke-RestMethod -Method Post -Uri http://localhost/predict -ContentType 'application/json' -InFile sample_request.json"
Start-Process "http://localhost/docs"
Next

Step 9 "Monitoring - Prometheus + Grafana stack, generate traffic"
Show "docker compose -f monitoring/docker-compose.yml up -d --build"
Wait-Url "http://localhost:8000/health"
Wait-Url "http://localhost:3000/api/health"
Start-Process "http://localhost:3000"
Start-Process "http://localhost:9090/targets"
Show "$py scripts/generate_traffic.py --n 300 --delay 0.1"
Write-Host "Show: Grafana dashboard panels filling up; Prometheus target UP."
Next

Step 10 "Request logging - structured JSON logs"
Show "docker logs heart-disease-monitoring-api-1 --tail 6"
Show "kubectl -n heart-disease logs deploy/heart-disease-api --tail 4"
Next

Step 11 "Alerting - stop the API and watch ApiDown fire"
Start-Process "http://localhost:9090/alerts"
Show "docker stop heart-disease-monitoring-api-1"
$last = ""
for ($i = 0; $i -lt 120; $i += 3) {
    $state = "inactive"
    try {
        $alerts = (Invoke-RestMethod http://localhost:9090/api/v1/alerts).data.alerts | Where-Object { $_.labels.alertname -eq "ApiDown" }
        if ($alerts) { $state = $alerts[0].state }
    } catch {}
    if ($state -ne $last) {
        Write-Host ("[{0,3}s] ApiDown = {1}   -> refresh the Alerts page now" -f $i, $state.ToUpper()) -ForegroundColor Magenta
        $last = $state
    }
    if ($state -eq "firing") { break }
    Start-Sleep 3
}
Write-Host "Show the FIRING alert in Prometheus and the red 'API up' panel in Grafana."
Next
Show "docker start heart-disease-monitoring-api-1"
Write-Host "API restored - the alert clears within about a minute."

Step 12 "Done - full pipeline demonstrated"
Write-Host "Data -> EDA/Training -> MLflow -> CI/CD -> Docker -> Kubernetes -> Monitoring & Alerts" -ForegroundColor Green
