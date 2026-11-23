$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
function Assert($condition, $message) { if (-not $condition) { throw $message } }
$desktop = Get-Content -Raw (Join-Path $PSScriptRoot 'texelith.desktop')
$control = Get-Content -Raw (Join-Path $PSScriptRoot 'control.in')
[xml]$svg = Get-Content -Raw (Join-Path $PSScriptRoot 'texelith.svg')
Assert ($desktop -match '(?m)^\[Desktop Entry\]\r?$') 'Missing desktop section'
foreach ($line in @('Type=Application','Name=Texelith','Exec=/opt/texelith/texelith-launch %f','Icon=texelith','Terminal=false','MimeType=image/png;image/jpeg;')) {
    Assert ($desktop -match ('(?m)^' + [regex]::Escape($line) + '\r?$')) "Invalid desktop field: $line"
}
Assert ($svg.DocumentElement.LocalName -eq 'svg' -and $svg.DocumentElement.NamespaceURI -eq 'http://www.w3.org/2000/svg') 'Invalid SVG root'
foreach ($field in @('Package: texelith','Version: @VERSION@','Architecture: amd64','Depends: @DEPENDS@','Installed-Size: @INSTALLED_SIZE@','Description:')) {
    Assert ($control.Contains($field)) "Missing control field: $field"
}
foreach ($path in @('LICENSE','THIRD_PARTY_NOTICES.md','crates/texelith-ui-egui/Cargo.toml','scripts/package-linux.sh','packaging/linux/texelith-launch')) {
    Assert (Test-Path -LiteralPath (Join-Path $root $path) -PathType Leaf) "Missing required path: $path"
}
$script = Get-Content -Raw (Join-Path $root 'scripts/package-linux.sh')
Assert ($script -match 'usr/share/applications') 'Missing package resource paths'
$dependencyLine = [regex]::Match($script, '(?m)^deps="\$deps, ([^"\r\n]+)"\r?$')
Assert ($dependencyLine.Success) 'Missing explicit runtime dependency assignment'
$runtimeDependencies = @($dependencyLine.Groups[1].Value -split ',\s*')
foreach ($dependency in @('libvulkan1','libwayland-client0','libxkbcommon0','libx11-6','libxcursor1','libxi6','libx11-xcb1','libxkbcommon-x11-0')) {
    Assert ($runtimeDependencies -ccontains $dependency) "Missing explicit runtime dependency: $dependency"
}
Write-Host 'Linux package metadata and source paths: PASS (static only)'
