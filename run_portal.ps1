Write-Host "Starting Axiom Dispatch Configuration Portal on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
uv run python -m app.web.server $args

