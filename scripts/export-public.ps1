[CmdletBinding()]
param([Parameter(Mandatory)][string]$Out)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent)).TrimEnd([IO.Path]::DirectorySeparatorChar)
$destination = [IO.Path]::GetFullPath($Out).TrimEnd([IO.Path]::DirectorySeparatorChar)
if ($destination -eq $root -or $destination.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must be outside the source tree.' }
if (Test-Path -LiteralPath $destination) {
    if (-not (Test-Path -LiteralPath $destination -PathType Container) -or @(Get-ChildItem -Force -LiteralPath $destination).Count) { throw 'Output must be a new or empty directory.' }
} else { New-Item -ItemType Directory -Path $destination | Out-Null }
$paths = @('crates', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'deny.toml', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'README.md', 'ROADMAP.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md', 'SECURITY.md', 'docs/adr', 'docs/architecture.md', 'docs/ci.md', 'docs/packaging.md', 'docs/licensing.md', 'docs/public-release.md', 'docs/images', 'scripts', 'packaging', '.github', '.gitignore', '.gitattributes')
foreach ($path in $paths) {
    $source = Join-Path $root $path
    if (-not (Test-Path -LiteralPath $source)) { Write-Warning "Missing optional release input: $path"; continue }
    $items = if (Test-Path -LiteralPath $source -PathType Container) { Get-ChildItem -LiteralPath $source -Recurse -File -Force } else { Get-Item -LiteralPath $source }
    foreach ($item in $items) {
        $relative = $item.FullName.Substring($root.Length + 1)
        if ($relative -match '(?i)(^|[\\/])(target|target-evidence|\.git)([\\/]|$)' -or $item.Extension -ieq '.log') { continue }
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing linked file: $relative" }
        $target = Join-Path $destination $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
        Copy-Item -LiteralPath $item.FullName -Destination $target
    }
}
# Assemble sensitive markers to keep the checker from matching its own source.
# The maintainer's public GitHub handle is allowed (repository URLs); the local profile name is not.
$markers = @(('far' + 'uk'), ('Codex' + 'Bar'), ('.orches' + 'tra'))
$credentialPrefixes = @(('sk' + '-'), ('gh' + 'p_'), ('AK' + 'IA'))
$failed = $false
$files = @(Get-ChildItem -LiteralPath $destination -Recurse -File -Force)
foreach ($file in $files) {
    $relative = $file.FullName.Substring($destination.Length + 1)
    if ($file.Length -gt 1MB -and $relative -ine 'THIRD_PARTY_NOTICES.md') { Write-Output "Large file: $relative ($($file.Length) bytes)" }
    $lines = [IO.File]::ReadAllLines($file.FullName)
    for ($i = 0; $i -lt $lines.Length; $i++) {
        $line = $lines[$i]
        $bad = $line -match '(?i)[a-z]:[\\/]users[\\/][^\\/\s"'']+'
        foreach ($marker in $markers) { if ($line.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) { $bad = $true } }
        foreach ($prefix in $credentialPrefixes) { if ($line -cmatch ('(?<![A-Za-z0-9_])' + [regex]::Escape($prefix) + '[A-Za-z0-9_-]{8,}')) { $bad = $true } }
        foreach ($email in [regex]::Matches($line, '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')) {
            $copyrightNoticeEmail = $relative -ieq 'THIRD_PARTY_NOTICES.md' -and $line -match '(?i)Copyright'
            if (-not $copyrightNoticeEmail -and $email.Value -notmatch '(?i)(@users\.noreply\.github\.com|^noreply@|@(?:[a-z0-9-]+\.)*example\.(?:com|org|net))$' -and $email.Value -notmatch '(?i)^noreply@') { $bad = $true }
        }
        # Long mixed-case base64-like values with digits and punctuation are plausible keys.
        foreach ($token in [regex]::Matches($line, '(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{32,}(?![A-Za-z0-9+/=_-])')) {
            $value = $token.Value
            $upper = [regex]::Matches($value, '[A-Z]').Count
            $digits = [regex]::Matches($value, '[0-9]').Count
            if ($value -cnotmatch '[a-z]' -or $upper -lt 3 -or $digits -lt 3 -or $value -match '^[a-fA-F0-9]+$') { continue }
            $entropy = 0.0
            foreach ($group in ($value.ToCharArray() | Group-Object -CaseSensitive)) {
                $p = $group.Count / $value.Length
                $entropy -= $p * [Math]::Log($p, 2)
            }
            if ($entropy -ge 4.5) { $bad = $true }
        }
        if ($bad) { Write-Output "${relative}:$($i + 1): sensitive content"; $failed = $true }
    }
}
$bytes = ($files | Measure-Object -Property Length -Sum).Sum
Write-Output "Export: $($files.Count) files, $bytes bytes"
if ($failed) { throw 'Scrub failed.' }
Write-Output 'Scrub: clean'
