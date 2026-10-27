param([switch]$Offline)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$originalDirectory = Get-Location
$process = [System.Diagnostics.Process]::GetCurrentProcess()
$originalPriority = $process.PriorityClass
$originalTargetDirectory = $env:CARGO_TARGET_DIR
$logDirectory = Join-Path $projectRoot ('work/validation/' + [guid]::NewGuid().ToString('N'))
$cargoMode = @('--locked')
if ($Offline) { $cargoMode += '--offline' }

function Invoke-Checked {
    param([string]$Name, [string]$Program, [string[]]$Arguments)
    $logPath = Join-Path $logDirectory ($Name + '.log')
    $executable = @(Get-Command $Program -CommandType Application -ErrorAction Stop)[0].Source
    $savedPreference = $ErrorActionPreference
    Write-Output "Starting $Name; log: $logPath"
    try {
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = $null
        & $executable @Arguments *> $logPath
        $exitCode = $global:LASTEXITCODE
    }
    finally { $ErrorActionPreference = $savedPreference }
    if ($null -eq $exitCode -or $exitCode -ne 0) {
        throw "$Name did not complete successfully (exit=$exitCode); see $logPath"
    }
    Write-Output "$Name passed"
}

try {
    Set-Location -LiteralPath $projectRoot
    $env:CARGO_TARGET_DIR = Join-Path $projectRoot 'work/target'
    $process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    New-Item -ItemType Directory -Path $logDirectory -ErrorAction Stop | Out-Null
    Invoke-Checked 'structure' 'python' @('scripts/check_structure.py')
    Invoke-Checked 'format' 'cargo' @('fmt', '--all', '--check')
    Invoke-Checked 'clippy' 'cargo' (@('clippy') + $cargoMode + @('--all-targets', '--', '-D', 'warnings'))
    Invoke-Checked 'rust-tests' 'cargo' (@('test') + $cargoMode + @('--', '--test-threads=1'))
    Invoke-Checked 'python-tests' 'python' @('-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v')
    foreach ($profile in @('tiny', 'dev-81m', 'local-287m')) {
        Invoke-Checked ('plan-' + $profile) 'cargo' (@('run') + $cargoMode + @('--', 'plan', '--config', "configs/$profile.json"))
    }
    Invoke-Checked 'smoke' 'cargo' (@('run') + $cargoMode + @('--', 'smoke', '--steps', '4', '--seed', '7'))
    Write-Output "Validation completed. Logs: $logDirectory"
}
finally {
    $process.PriorityClass = $originalPriority
    $env:CARGO_TARGET_DIR = $originalTargetDirectory
    Set-Location -LiteralPath $originalDirectory
}
