$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$clientRoot = if ((Split-Path -Leaf $scriptDir) -eq "scripts") {
    Split-Path -Parent $scriptDir
} else {
    $scriptDir
}
$xAnyLabeling = Join-Path $clientRoot ".venv\Scripts\xanylabeling.exe"
$sourceCatalog = Join-Path $clientRoot "anylabeling\configs\models.yaml"
$packagedCatalog = Join-Path $clientRoot "remote-models.yaml"
$installedCatalog = Join-Path $clientRoot ".venv\Lib\site-packages\anylabeling\configs\models.yaml"
$clientConfig = Join-Path $clientRoot ".xanylabelingrc"
$defaultConfig = Join-Path $clientRoot "anylabeling\configs\xanylabeling_config.yaml"
$secretFile = Join-Path $clientRoot "secrets\api_key"
$modelUploadSecretFile = Join-Path $clientRoot "secrets\model_upload_key"

$serverUrl = if ($env:LANGGAO_AUTOLABEL_URL) {
    $env:LANGGAO_AUTOLABEL_URL.TrimEnd("/")
} else {
    "http://127.0.0.1:18618"
}
$sshTarget = if ($env:LANGGAO_AUTOLABEL_SSH) {
    $env:LANGGAO_AUTOLABEL_SSH
} else {
    "4090"
}

if (-not (Test-Path -LiteralPath $xAnyLabeling)) {
    throw "X-AnyLabeling client environment not found: $xAnyLabeling"
}

if (-not (Test-Path -LiteralPath $clientConfig)) {
    Copy-Item -LiteralPath $defaultConfig -Destination $clientConfig
}

$configText = [IO.File]::ReadAllText($clientConfig)
$existingKey = [regex]::Match(
    $configText,
    '(?m)^\s*api_key:\s*["'']?([^\r\n"'']+)'
).Groups[1].Value.Trim()

$apiKey = $env:LANGGAO_AUTOLABEL_API_KEY
if (-not $apiKey -and (Test-Path -LiteralPath $secretFile)) {
    $apiKey = [IO.File]::ReadAllText($secretFile).Trim()
}
if (-not $apiKey -and $existingKey -and $existingKey -ne "null") {
    $apiKey = $existingKey
}
if (-not $apiKey) {
    throw "Set LANGGAO_AUTOLABEL_API_KEY or create secrets\api_key."
}

$remoteBlock = @"
remote_server_settings:
  server_url: $serverUrl
  api_key: null
  timeout: 300
"@
$configText = [regex]::Replace(
    $configText,
    '(?ms)^remote_server_settings:\s*\r?\n(?:[ \t]+.*(?:\r?\n|$))*',
    $remoteBlock + [Environment]::NewLine
)
$configText = [regex]::Replace(
    $configText,
    '(?m)^model_hub:\s*.*$',
    'model_hub: modelscope'
)
[IO.File]::WriteAllText(
    $clientConfig,
    $configText,
    [Text.UTF8Encoding]::new($false)
)

# Keep the installed package in remote-only mode even when launched directly.
if (Test-Path -LiteralPath $sourceCatalog) {
    Copy-Item -LiteralPath $sourceCatalog -Destination $installedCatalog -Force
}
elseif (Test-Path -LiteralPath $packagedCatalog) {
    Copy-Item -LiteralPath $packagedCatalog -Destination $installedCatalog -Force
}
elseif (-not (Test-Path -LiteralPath $installedCatalog)) {
    throw "Remote-only model catalog is missing."
}

$env:XANYLABELING_REMOTE_ONLY = "1"
$env:XANYLABELING_SERVER_URL = $serverUrl
$env:XANYLABELING_SERVER_API_KEY = $apiKey
if (Test-Path -LiteralPath $modelUploadSecretFile) {
    $env:XANYLABELING_MODEL_UPLOAD_API_KEY = (
        [IO.File]::ReadAllText($modelUploadSecretFile).Trim()
    )
}

function Test-AutoLabelHealth {
    try {
        $health = Invoke-RestMethod -Uri "$serverUrl/health" -TimeoutSec 3
        return $health.status -eq "healthy"
    }
    catch {
        return $false
    }
}

if (-not (Test-AutoLabelHealth)) {
    $uri = [Uri]$serverUrl
    if ($uri.Host -notin @("127.0.0.1", "localhost")) {
        throw "Remote server is unavailable: $serverUrl"
    }

    $sshArgs = @(
        "-N",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-L", "$($uri.Port):127.0.0.1:$($uri.Port)",
        $sshTarget
    )
    Start-Process -FilePath "ssh.exe" -ArgumentList $sshArgs -WindowStyle Hidden

    $ready = $false
    foreach ($attempt in 1..20) {
        Start-Sleep -Seconds 1
        if (Test-AutoLabelHealth) {
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        throw "Unable to connect to $sshTarget or start the SSH tunnel."
    }
}

Start-Process `
    -FilePath $xAnyLabeling `
    -ArgumentList @("--work-dir", $clientRoot) `
    -WorkingDirectory $clientRoot
