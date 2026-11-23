#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mode=build
output=target-evidence/T-092/package
while (($#)); do
  case "$1" in
    --self-test) mode=self-test ;;
    --verify) mode=verify ;;
    --output) shift; output="${1:?missing output path}" ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Requires native Apple Silicon macOS' >&2; exit 1; }
for file in packaging/macos/Info.plist packaging/macos/icon.png LICENSE THIRD_PARTY_NOTICES.md; do
  [[ -f "$file" ]] || { echo "Missing $file" >&2; exit 1; }
done
plutil -lint packaging/macos/Info.plist
[[ "$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' packaging/macos/Info.plist)" == 13.0 ]]
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' packaging/macos/Info.plist)" == texelith-ui-egui ]]
if [[ "$mode" == self-test ]]; then echo 'PASS: templates and paths; no DMG or runtime tested'; exit 0; fi
export MACOSX_DEPLOYMENT_TARGET=13.0
cargo build --release --locked -p texelith-ui-egui --target aarch64-apple-darwin
binary=target/aarch64-apple-darwin/release/texelith-ui-egui
[[ "$(lipo -archs "$binary")" == arm64 ]]
minos="$(otool -l "$binary" | awk '$1 == "minos" {print $2; exit}')"
[[ "$minos" == 13.0 || "$minos" == 13.0.0 ]] || { echo "Unexpected Mach-O minimum: $minos" >&2; exit 1; }
# A standalone bundle must not depend on Homebrew or build-machine libraries.
otool -L "$binary" | tail -n +2 | while IFS= read -r line; do
  lib="${line#${line%%[![:space:]]*}}"; lib="${lib%% *}"
  case "$lib" in /System/Library/*|/usr/lib/*) ;; *) echo "Unbundled library: $lib" >&2; exit 1 ;; esac
done
mkdir -p "$output"
output="$(cd "$output" && pwd)"
{ sw_vers; xcode-select -p; xcrun --show-sdk-path; xcrun --show-sdk-version; rustc --version; cargo --version; otool -L "$binary"; otool -l "$binary"; } > "$output/build-inventory.txt"
work="$(mktemp -d "${TMPDIR:-/tmp}/texelith-package.XXXXXX")"
mount=""
cleanup() { if [[ -n "$mount" ]]; then hdiutil detach "$mount" >/dev/null || true; fi; rm -rf "$work"; }
trap cleanup EXIT
app="$work/dmg/Texelith.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$work/Texelith.iconset"
cp "$binary" "$app/Contents/MacOS/texelith-ui-egui"
cp packaging/macos/Info.plist "$app/Contents/Info.plist"
cp LICENSE THIRD_PARTY_NOTICES.md "$app/Contents/Resources/"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" packaging/macos/icon.png --out "$work/Texelith.iconset/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" packaging/macos/icon.png --out "$work/Texelith.iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$work/Texelith.iconset" -o "$app/Contents/Resources/Texelith.icns"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
ln -s /Applications "$work/dmg/Applications"
dmg="$output/Texelith-macos-arm64.dmg"
rm -f "$dmg"
hdiutil create -quiet -srcfolder "$work/dmg" -volname Texelith -format UDZO "$dmg"
shasum -a 256 "$dmg" > "$dmg.sha256"
archive="$output/Texelith-macos-arm64.zip"
ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
shasum -a 256 "$archive" > "$archive.sha256"
(( $(stat -f %z "$archive") < 50000000 )) || { echo 'ZIP size target missed' >&2; exit 1; }
size="$(stat -f %z "$dmg")"
echo "DMG bytes: $size (target <50000000); installed bytes: $(du -sk "$app" | awk '{print $1 * 1024}')"
if ((size >= 50000000)); then echo 'DMG size target missed' >&2; exit 1; fi
if [[ "$mode" == verify ]]; then
  mkdir "$work/mount" "$work/install"
  hdiutil attach -quiet -nobrowse -readonly -mountpoint "$work/mount" "$dmg"
  mount="$work/mount"
  ditto "$mount/Texelith.app" "$work/install/Texelith.app"
  hdiutil detach -quiet "$mount"; mount=""
  codesign --verify --deep --strict "$work/install/Texelith.app"
  "$work/install/Texelith.app/Contents/MacOS/texelith-ui-egui" > "$output/launch.log" 2>&1 &
  pid=$!
  sleep 10
  if ! kill -0 "$pid" 2>/dev/null; then wait "$pid" || true; echo 'Application exited during launch check' >&2; exit 1; fi
  kill "$pid"; wait "$pid" || true
  rm -rf "$work/install/Texelith.app"
  [[ ! -e "$work/install/Texelith.app" ]]
  echo 'PASS: DMG attach, copy, process launch and removal; visual interaction requires manual verification'
fi
