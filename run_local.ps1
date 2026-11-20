param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.local.toml'),
    [string]$Python = 'python',
    [switch]$Refit,
    [switch]$LegacyUserSite
)
$ErrorActionPreference = 'Stop'
$runArguments = @()
if ($LegacyUserSite) { $runArguments += '-s' }
$runArguments += @((Join-Path $PSScriptRoot 'run.py'), '--config', $Config)
if ($LegacyUserSite) { $runArguments += '--legacy-user-site' }
$runArguments += 'run'
if ($Refit) { $runArguments += '--refit' }
& $Python @runArguments
if ($LASTEXITCODE -ne 0) { throw "Reconstruction failed (exit $LASTEXITCODE)" }
