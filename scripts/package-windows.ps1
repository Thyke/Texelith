param([switch]$SelfTest, [switch]$Verify, [string]$OutputDir = 'target-evidence/T-091/package')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$assets = Join-Path $repo 'packaging/windows'
function Assert($condition, $message) { if (!$condition) { throw $message } }
function Exercise([string]$payload, [string]$temp) {
    $root = Join-Path $temp 'User space é\Texelith'
    $programs = Join-Path $temp 'Start menu é'
    & (Join-Path $payload 'install.ps1') -InstallRoot $root -ProgramsRoot $programs
    foreach ($f in @('texelith-ui-egui.exe','LICENSE','THIRD_PARTY_NOTICES.md','uninstall.ps1','texelith.ico')) { Assert (Test-Path -LiteralPath (Join-Path $root $f) -PathType Leaf) "Missing installed file $f" }
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $programs 'Texelith.lnk'))
    Assert ($link.TargetPath -ieq (Join-Path $root 'texelith-ui-egui.exe')) 'Shortcut target mismatch'
    Set-Content -LiteralPath (Join-Path $root 'project.texelith') -Value 'user data'
    & (Join-Path $payload 'install.ps1') -InstallRoot $root -ProgramsRoot $programs
    Assert ((Get-Content -LiteralPath (Join-Path $root 'project.texelith') -Raw).Trim() -eq 'user data') 'Project changed during repeated install'
    & (Join-Path $payload 'uninstall.ps1') -InstallRoot $root -ProgramsRoot $programs
    Assert (Test-Path -LiteralPath (Join-Path $root 'project.texelith')) 'Project lost'
    Assert (!(Test-Path -LiteralPath (Join-Path $root 'texelith-ui-egui.exe'))) 'Executable remains'
    Assert (!(Test-Path -LiteralPath (Join-Path $programs 'Texelith.lnk'))) 'Shortcut remains'
    & (Join-Path $payload 'install.ps1') -InstallRoot $root -ProgramsRoot $programs
    Assert ((Get-Content -LiteralPath (Join-Path $root 'project.texelith') -Raw).Trim() -eq 'user data') 'Project changed after uninstall and reinstall'
    Assert (Test-Path -LiteralPath (Join-Path $root 'texelith-ui-egui.exe') -PathType Leaf) 'Reinstall failed'
    & (Join-Path $payload 'uninstall.ps1') -InstallRoot $root -ProgramsRoot $programs
}
$temp = Join-Path ([IO.Path]::GetTempPath()) ('Texelith-T091-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    if ($SelfTest) {
        $payload = Join-Path $temp 'payload'
        New-Item -ItemType Directory -Path $payload | Out-Null
        Copy-Item -Path (Join-Path $assets '*') -Destination $payload
        foreach ($f in @('texelith-ui-egui.exe','LICENSE','THIRD_PARTY_NOTICES.md')) { Set-Content -LiteralPath (Join-Path $payload $f) -Value 'fixture' }
        Exercise $payload (Join-Path $temp 'normal')
        $collision = Join-Path $temp 'collision'; New-Item -ItemType Directory $collision | Out-Null
        Set-Content -LiteralPath (Join-Path $collision 'LICENSE') -Value 'keep'
        $failed = $false
        try { & (Join-Path $payload 'install.ps1') -InstallRoot $collision -ProgramsRoot (Join-Path $temp 'links') } catch { $failed = $true }
        Assert $failed 'Directory collision did not fail'
        Assert (Test-Path -LiteralPath (Join-Path $collision 'LICENSE')) 'Collision data lost'
        $badPrograms = Join-Path $temp 'blocked'; Set-Content -LiteralPath $badPrograms -Value 'file'
        $rollback = Join-Path $temp 'rollback'
        $failed = $false
        try { & (Join-Path $payload 'install.ps1') -InstallRoot $rollback -ProgramsRoot $badPrograms } catch { $failed = $true }
        Assert $failed 'Injected failure did not fail'
        Assert (!(Test-Path -LiteralPath $rollback)) 'Failed install was not rolled back'
        $existing = Join-Path $temp 'existing'
        & (Join-Path $payload 'install.ps1') -InstallRoot $existing -ProgramsRoot (Join-Path $temp 'existing-links')
        Set-Content -LiteralPath (Join-Path $existing 'LICENSE') -Value 'old version'
        $failed = $false
        try { & (Join-Path $payload 'install.ps1') -InstallRoot $existing -ProgramsRoot $badPrograms } catch { $failed = $true }
        Assert $failed 'Injected upgrade failure did not fail'
        Assert ((Get-Content -LiteralPath (Join-Path $existing 'LICENSE') -Raw).Trim() -eq 'old version') 'Upgrade rollback did not restore files'
        Write-Output 'PASS: isolated paths, repeated install, reinstall, shortcut, preservation, collision and rollback'
        return
    }
    $out = [IO.Path]::GetFullPath((Join-Path $repo $OutputDir))
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Push-Location $repo
    try { & cargo build --release --locked -p texelith-ui-egui; if ($LASTEXITCODE -ne 0) { throw 'Cargo build failed' } } finally { Pop-Location }
    $payload = Join-Path $temp 'Texelith'
    New-Item -ItemType Directory -Path $payload | Out-Null
    foreach ($f in @('install.ps1','uninstall.ps1','texelith.ico')) { Copy-Item -LiteralPath (Join-Path $assets $f) -Destination $payload }
    foreach ($f in @('LICENSE','THIRD_PARTY_NOTICES.md')) { Copy-Item -LiteralPath (Join-Path $repo $f) -Destination $payload }
    Copy-Item -LiteralPath (Join-Path $repo 'target/release/texelith-ui-egui.exe') -Destination $payload
    Copy-Item -LiteralPath (Join-Path $repo 'target/release/texelith-ui-egui.exe') -Destination (Join-Path $payload 'Texelith.exe')
    Set-Content -LiteralPath (Join-Path $payload 'README.txt') -Encoding UTF8 -Value 'Texelith 0.1.0 for Windows x64. Extract the archive and run Texelith.exe, or run install.ps1 for a per-user installation. This build is unsigned. See LICENSE and THIRD_PARTY_NOTICES.md.'
    $zip = Join-Path $out 'Texelith-0.1.0-windows-x64.zip'
    Compress-Archive -Path (Join-Path $payload '*') -DestinationPath $zip -Force
    $size = (Get-Item -LiteralPath $zip).Length
    Assert ($size -lt 50000000) "ZIP exceeds 50 MB: $size bytes"
    Get-FileHash -LiteralPath $zip -Algorithm SHA256 | Format-List | Out-File (Join-Path $out 'SHA256.txt')
    Write-Output "ZIP: $zip ($size bytes); installed bytes: $((Get-ChildItem $payload -File | Measure-Object Length -Sum).Sum)"
    if ($Verify) {
        $unpacked = Join-Path $temp 'unpacked'; Expand-Archive -LiteralPath $zip -DestinationPath $unpacked
        Exercise $unpacked (Join-Path $temp 'verify')
        Write-Output 'PASS: ZIP unpack/install/shortcut/uninstall/project preservation'
    }
} finally { if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Recurse -Force } }


