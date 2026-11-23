$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
foreach ($path in @('packaging/macos/Info.plist', 'packaging/macos/icon.svg', 'packaging/macos/icon.png', 'scripts/package-macos.sh', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/ci/faz1-platforms.md')) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $path) -PathType Leaf)) { throw "Missing: $path" }
}
[xml]$plist = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'Info.plist')
$nodes = $plist.plist.dict.ChildNodes
$values = @{}
for ($i = 0; $i -lt $nodes.Count; $i += 2) {
    if ($nodes[$i].Name -ne 'key' -or $null -eq $nodes[$i + 1]) { throw 'Malformed plist dictionary' }
    $values[$nodes[$i].InnerText] = $nodes[$i + 1].InnerText
}
$expected = @{ CFBundleExecutable='texelith-ui-egui'; CFBundleIdentifier='com.texelith.app'; CFBundlePackageType='APPL'; CFBundleIconFile='Texelith.icns'; LSMinimumSystemVersion='13.0' }
foreach ($key in $expected.Keys) { if ($values[$key] -ne $expected[$key]) { throw "Unexpected $key" } }
[xml]$svg = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'icon.svg')
if ($svg.svg.viewBox -ne '0 0 1024 1024') { throw 'Unexpected icon dimensions' }
$script = Get-Content -Raw -LiteralPath (Join-Path $root 'scripts/package-macos.sh')
foreach ($token in @('--self-test', '--verify', 'hdiutil', 'MACOSX_DEPLOYMENT_TARGET', 'THIRD_PARTY_NOTICES.md', 'otool', 'lipo', 'iconutil', 'xcode-select -p', 'xcrun --show-sdk-path', 'xcrun --show-sdk-version')) {
    if (-not $script.Contains($token)) { throw "Missing script contract: $token" }
}
if ($script -match '(?m)^\s*[^#\r\n]*\bxcodebuild\b') { throw 'Packaging must not require full Xcode for toolchain inventory' }
Write-Host 'PASS: macOS metadata, templates and paths (no native runtime verification)'
