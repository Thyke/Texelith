# Build prerequisites only; no display server or GPU is required for CPU CI.
$ErrorActionPreference = 'Stop'
sudo apt-get update
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$packages = @(
    'build-essential', 'pkg-config', 'libx11-dev', 'libx11-xcb-dev',
    'libxcb1-dev', 'libxcb-render0-dev', 'libxcb-shape0-dev', 'libxcb-xfixes0-dev',
    'libxcursor-dev', 'libxi-dev', 'libxrandr-dev', 'libxinerama-dev',
    'libxkbcommon-dev', 'libxkbcommon-x11-dev', 'libwayland-dev', 'wayland-protocols',
    'libudev-dev', 'libegl1-mesa-dev', 'libvulkan-dev'
)
sudo apt-get install -y @packages
if ($LASTEXITCODE) { exit $LASTEXITCODE }
dpkg-query -W @packages
if ($LASTEXITCODE) { exit $LASTEXITCODE }
