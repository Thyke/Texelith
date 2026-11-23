#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
mode=build
out=target-evidence/T-093/package
while (($#)); do
  case "$1" in
    --self-test) mode=self-test ;;
    --verify) mode=verify ;;
    --output) shift; (($#)) || { echo 'Missing --output value' >&2; exit 2; }; out=$1 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Requires native Linux x86_64' >&2; exit 1; }
for tool in dpkg-deb dpkg dpkg-query dpkg-shlibdeps fakeroot readelf ldd cargo python3; do
  command -v "$tool" >/dev/null || { echo "Missing build prerequisite: $tool" >&2; exit 1; }
done
python3 - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET
p = Path('packaging/linux')
d = (p / 'texelith.desktop').read_text(encoding='utf-8')
c = (p / 'control.in').read_text(encoding='utf-8')
for line in ('[Desktop Entry]', 'Type=Application', 'Name=Texelith', 'Exec=/opt/texelith/texelith-launch %f', 'Icon=texelith', 'Terminal=false', 'MimeType=image/png;image/jpeg;'):
    assert line in d.splitlines(), line
assert ET.parse(p / 'texelith.svg').getroot().tag == '{http://www.w3.org/2000/svg}svg'
for line in ('Package: texelith', 'Version: @VERSION@', 'Architecture: amd64', 'Depends: @DEPENDS@', 'Installed-Size: @INSTALLED_SIZE@'):
    assert line in c.splitlines(), line
for path in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'crates/texelith-ui-egui/Cargo.toml'):
    assert Path(path).is_file(), path
PY
if [[ $mode == self-test ]]; then echo 'Linux resource and control self-test: PASS'; exit 0; fi
mkdir -p "$out"
out=$(cd "$out" && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cargo build --release --locked -p texelith-ui-egui --target x86_64-unknown-linux-gnu
binary=target/x86_64-unknown-linux-gnu/release/texelith-ui-egui
readelf --version-info "$binary" > "$out/readelf-versions.txt"
readelf -d "$binary" > "$out/readelf-dynamic.txt"
ldd "$binary" > "$out/ldd.txt"
if grep -q 'not found' "$out/ldd.txt"; then echo 'Unresolved shared libraries' >&2; exit 1; fi
python3 - "$out/readelf-versions.txt" <<'PY'
import re, sys
s = open(sys.argv[1], encoding='utf-8').read()
versions = [tuple(map(int, x.split('.'))) for x in re.findall(r'\bGLIBC_(\d+(?:\.\d+)+)', s)]
if not versions or max(versions) > (2, 35):
    raise SystemExit(f'glibc floor 2.35 exceeded or unavailable: {max(versions) if versions else "none"}')
PY
mkdir -p "$stage/DEBIAN" "$stage/opt/texelith" "$stage/usr/share/applications" "$stage/usr/share/icons/hicolor/scalable/apps" "$stage/usr/share/doc/texelith" "$stage/debian"
install -m755 "$binary" "$stage/opt/texelith/texelith-ui-egui"
install -m755 packaging/linux/texelith-launch "$stage/opt/texelith/texelith-launch"
install -m644 packaging/linux/texelith.desktop "$stage/usr/share/applications/texelith.desktop"
install -m644 packaging/linux/texelith.svg "$stage/usr/share/icons/hicolor/scalable/apps/texelith.svg"
install -m644 LICENSE THIRD_PARTY_NOTICES.md "$stage/usr/share/doc/texelith/"
# dpkg-shlibdeps uses the host distribution's package database to generate dependencies.
printf 'Source: texelith\nSection: graphics\nPriority: optional\nMaintainer: Texelith contributors\nStandards-Version: 4.6.0\n\nPackage: texelith\nArchitecture: amd64\nDescription: Texelith\n' > "$stage/debian/control"
deps=$(cd "$stage" && dpkg-shlibdeps -O opt/texelith/texelith-ui-egui | sed -n 's/^shlibs:Depends=//p')
[[ -n $deps ]] || { echo 'No generated runtime dependencies' >&2; exit 1; }
deps="$deps, libvulkan1, libwayland-client0, libxkbcommon0, libx11-6, libxcursor1, libxi6, libx11-xcb1, libxkbcommon-x11-0"
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' crates/texelith-ui-egui/Cargo.toml | head -n1)
[[ $version =~ ^[0-9][0-9A-Za-z.+~:-]*$ ]] || { echo 'Invalid package version' >&2; exit 1; }
size=$(du -sk "$stage/opt" "$stage/usr" | awk '{s += $1} END {print s}')
VERSION="$version" DEPENDS="$deps" INSTALLED_SIZE="$size" python3 - "$stage/DEBIAN/control" <<'PY'
import os, pathlib, sys
s = pathlib.Path('packaging/linux/control.in').read_text(encoding='utf-8')
for k in ('VERSION', 'DEPENDS', 'INSTALLED_SIZE'):
    s = s.replace('@' + k + '@', os.environ[k])
pathlib.Path(sys.argv[1]).write_text(s, encoding='utf-8')
PY
rm -rf "$stage/debian"
package="$out/texelith_${version}_amd64.deb"
fakeroot dpkg-deb --build --root-owner-group "$stage" "$package"
dpkg-deb --info "$package" > "$out/package-info.txt"
dpkg-deb --contents "$package" > "$out/package-contents.txt"
sha256sum "$package" > "$out/SHA256SUMS"
archive="$out/Texelith-linux-x86_64.tar.gz"
tar -czf "$archive" -C "$stage" opt usr
sha256sum "$archive" >> "$out/SHA256SUMS"
(( $(stat -c %s "$archive") < 50000000 )) || { echo 'Archive exceeds 50 MB target' >&2; exit 1; }
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'; uname -a; cargo --version; rustc --version
  git rev-parse HEAD
  dpkg-query -W dpkg dpkg-dev fakeroot binutils python3
  dpkg-deb --version | head -n1; fakeroot --version | head -n1
  printf 'dpkg: GPL-2.0-or-later, build-only; fakeroot: GPL-3.0-or-later, build-only; neither bundled.\n'
  printf 'Installed size KiB: %s\nCompressed size bytes: %s\n' "$size" "$(stat -c %s "$package")"
} > "$out/build-inventory.txt"
(( $(stat -c %s "$package") < 50000000 )) || { echo 'Package exceeds 50 MB target' >&2; exit 1; }
if [[ $mode == verify ]]; then
  # Simulation checks actual host package dependency closure without changing host state.
  dpkg --simulate --install "$package" > "$out/dependency-simulation.txt" 2>&1
  isolated=$(mktemp -d)
  trap 'rm -rf "$stage" "$isolated"' EXIT
  mkdir -p "$isolated/var/lib/dpkg"
  : > "$isolated/var/lib/dpkg/status"
  # Closure was checked above; force-depends is needed for the empty isolated database.
  fakeroot dpkg --root="$isolated" --force-depends --install "$package" > "$out/install.txt" 2>&1
  test -x "$isolated/opt/texelith/texelith-ui-egui"
  mkdir -p "$isolated/home/user"
  printf 'preserve\n' > "$isolated/home/user/project.texelith"
  # Requires an available native graphical session and Vulkan adapter.
  HOME="$isolated/home/user" timeout 60 "$isolated/opt/texelith/texelith-ui-egui" --render-smoke --backend vulkan --frames 3 --report "$out/launch.json" > "$out/launch.log" 2>&1
  fakeroot dpkg --root="$isolated" --remove texelith > "$out/remove.txt" 2>&1
  test ! -e "$isolated/opt/texelith/texelith-ui-egui"
  test "$(cat "$isolated/home/user/project.texelith")" = preserve
  echo 'Native isolated install, launch and removal: PASS'
fi
