param(
    [string]$InstallRoot = $PSScriptRoot,
    [string]$ProgramsRoot = (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
)
$ErrorActionPreference = 'Stop'
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$shortcut = Join-Path ([IO.Path]::GetFullPath($ProgramsRoot)) 'Texelith.lnk'
if ((Test-Path -LiteralPath $shortcut -PathType Leaf)) {
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
    if ($link.TargetPath -ieq (Join-Path $InstallRoot 'texelith-ui-egui.exe')) { Remove-Item -LiteralPath $shortcut -Force }
}
foreach ($file in @('texelith-ui-egui.exe', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'uninstall.ps1', 'texelith.ico', '.texelith-install')) {
    $path = Join-Path $InstallRoot $file
    if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path -Force }
}
if ((Test-Path -LiteralPath $InstallRoot -PathType Container) -and !(Get-ChildItem -LiteralPath $InstallRoot -Force | Select-Object -First 1)) { Remove-Item -LiteralPath $InstallRoot -Force }
Write-Output 'Uninstalled Texelith; other files were preserved.'

