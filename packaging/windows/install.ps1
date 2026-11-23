param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'Programs\Texelith'),
    [string]$ProgramsRoot = (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
)
$ErrorActionPreference = 'Stop'
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$ProgramsRoot = [IO.Path]::GetFullPath($ProgramsRoot)
$shortcut = Join-Path $ProgramsRoot 'Texelith.lnk'
$files = @('texelith-ui-egui.exe', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'uninstall.ps1', 'texelith.ico')
foreach ($file in $files) { if (!(Test-Path -LiteralPath (Join-Path $PSScriptRoot $file) -PathType Leaf)) { throw "Missing package file: $file" } }
if ((Test-Path -LiteralPath $InstallRoot) -and !(Test-Path -LiteralPath $InstallRoot -PathType Container)) { throw 'Install path is not a directory.' }
$marker = Join-Path $InstallRoot '.texelith-install'
$owned = (Test-Path -LiteralPath $marker -PathType Leaf) -and ((Get-Content -LiteralPath $marker -Raw).Trim() -ceq 'Texelith per-user installation v1')
# Recognize installations created by the previous script, without adopting arbitrary files.
if (!$owned -and (Test-Path -LiteralPath (Join-Path $InstallRoot 'uninstall.ps1') -PathType Leaf)) {
    $oldUninstaller = (Get-Content -LiteralPath (Join-Path $InstallRoot 'uninstall.ps1') -Raw).Trim() -replace "`r`n", "`n"
    $legacyUninstaller = ((Get-Content -LiteralPath (Join-Path $PSScriptRoot 'uninstall.ps1') -Raw).Replace(", '.texelith-install'", '')).Trim() -replace "`r`n", "`n"
    $owned = ($oldUninstaller -ceq $legacyUninstaller) -and
        (@($files | Where-Object { !(Test-Path -LiteralPath (Join-Path $InstallRoot $_) -PathType Leaf) }).Count -eq 0)
}
if (!$owned) {
    foreach ($file in ($files + '.texelith-install')) { if (Test-Path -LiteralPath (Join-Path $InstallRoot $file)) { throw "Install file collision: $file" } }
}
if (Test-Path -LiteralPath $shortcut) {
    if (!(Test-Path -LiteralPath $shortcut -PathType Leaf)) { throw 'Shortcut path collision.' }
    $existingLink = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
    if ($existingLink.TargetPath -ine (Join-Path $InstallRoot 'texelith-ui-egui.exe')) { throw 'Shortcut belongs to another installation.' }
}
$backup = Join-Path ([IO.Path]::GetTempPath()) ('Texelith-install-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $backup | Out-Null
$rootExisted = Test-Path -LiteralPath $InstallRoot -PathType Container
$shortcutExisted = Test-Path -LiteralPath $shortcut -PathType Leaf
$changed = @()
try {
    if ($shortcutExisted) { Copy-Item -LiteralPath $shortcut -Destination (Join-Path $backup 'shortcut') }
    foreach ($file in ($files + '.texelith-install')) {
        $destination = Join-Path $InstallRoot $file
        if (Test-Path -LiteralPath $destination) {
            if (!(Test-Path -LiteralPath $destination -PathType Leaf)) { throw "Install file collision: $file" }
            Copy-Item -LiteralPath $destination -Destination (Join-Path $backup $file)
        }
    }
    New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
    foreach ($file in $files) {
        $changed += $file
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $file) -Destination (Join-Path $InstallRoot $file) -Force
    }
    $changed += '.texelith-install'
    Set-Content -LiteralPath $marker -Value 'Texelith per-user installation v1'
    if (!(Test-Path -LiteralPath $ProgramsRoot -PathType Container)) { New-Item -ItemType Directory -Path $ProgramsRoot -Force | Out-Null }
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
    $link.TargetPath = Join-Path $InstallRoot 'texelith-ui-egui.exe'
    $link.WorkingDirectory = $InstallRoot
    $link.IconLocation = (Join-Path $InstallRoot 'texelith.ico') + ',0'
    $link.Save()
    Write-Output "Installed Texelith to $InstallRoot"
} catch {
    foreach ($file in $changed) {
        $saved = Join-Path $backup $file
        $destination = Join-Path $InstallRoot $file
        if (Test-Path -LiteralPath $saved -PathType Leaf) { Copy-Item -LiteralPath $saved -Destination $destination -Force -ErrorAction SilentlyContinue }
        elseif (Test-Path -LiteralPath $destination -PathType Leaf) { Remove-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue }
    }
    if ($shortcutExisted) { Copy-Item -LiteralPath (Join-Path $backup 'shortcut') -Destination $shortcut -Force -ErrorAction SilentlyContinue }
    elseif (Test-Path -LiteralPath $shortcut -PathType Leaf) { Remove-Item -LiteralPath $shortcut -Force -ErrorAction SilentlyContinue }
    if (!$rootExisted -and (Test-Path -LiteralPath $InstallRoot -PathType Container) -and !(Get-ChildItem -LiteralPath $InstallRoot -Force | Select-Object -First 1)) { Remove-Item -LiteralPath $InstallRoot -Force -ErrorAction SilentlyContinue }
    throw
} finally { Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction SilentlyContinue }

