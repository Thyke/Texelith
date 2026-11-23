param(
    [string]$InputDirectory = 'target/readme-inputs',
    [string]$CaptureDirectory = 'target-evidence/T-168',
    [switch]$SkipCapture
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
# Requires the repository Rust toolchain and Python with Pillow. No desktop capture tools.
$materials = [ordered]@{
    brick = 'red_brick_03_diff_2k.jpg'
    rock = 'rock_wall_08_diff_2k.jpg'
    wood = 'wood_planks_diff_2k.jpg'
    metal = 'rusty_metal_02_diff_2k.jpg'
}
if (-not $SkipCapture) {
    foreach ($file in $materials.Values) {
        if (-not (Test-Path -LiteralPath (Join-Path $InputDirectory $file))) {
            throw "Missing Poly Haven CC0 input: $file"
        }
    }
    cargo build -p texelith-ui-egui --locked
    if ($LASTEXITCODE -ne 0) { throw 'Snapshot build failed' }
    # Sequential captures share the GPU. egui captures its own surface at scale 1.0.
    $exe = Join-Path $(if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { 'target' }) 'debug/texelith-ui-egui.exe'
    New-Item -ItemType Directory -Force -Path $CaptureDirectory | Out-Null
    foreach ($entry in $materials.GetEnumerator()) {
        $out = Join-Path $CaptureDirectory $entry.Key
        # Start-Process joins arguments into one Windows command line; quote paths with spaces.
        $arguments = @('--ui-snapshot', "`"$out`"", '--readme-snapshots', '--scale', '1.0',
            '--input', "`"$(Join-Path $InputDirectory $entry.Value)`"", '--report', "`"$out.json`"")
        $process = Start-Process -FilePath (Resolve-Path $exe) -ArgumentList $arguments -WindowStyle Hidden `
            -RedirectStandardError "$out.log" -PassThru -Wait
        if ($process.ExitCode -ne 0) { throw "Snapshot failed for $($entry.Key); see $out.log" }
        $report = Get-Content -Raw -LiteralPath "$out.json" | ConvertFrom-Json
        if (-not $report.passed) { throw "Snapshot validation failed for $($entry.Key)" }
    }
}
@'
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

captures = Path(sys.argv[1])
output = Path('docs/images')
output.mkdir(parents=True, exist_ok=True)
fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
regular = lambda size: ImageFont.truetype(str(fonts / 'segoeui.ttf'), size)
bold = lambda size: ImageFont.truetype(str(fonts / 'segoeuib.ttf'), size)
bg, panel, white, muted, amber = '#17191c', '#222529', '#f2f3f5', '#adb4be', '#f5a524'

def capture(material, state):
    path = captures / material / f'studio-{state}-1920x1080-1.0.png'
    with Image.open(path) as source:
        if source.size != (1920, 1080):
            raise ValueError(f'{path}: expected native 1920x1080, got {source.size}')
        return source.convert('RGB')

def canvas(title, subtitle, height=1080):
    image = Image.new('RGB', (1920, height), bg)
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 34, 37, 78), fill=amber)
    draw.text((54, 25), title, font=bold(34), fill=white)
    draw.text((54, 75), subtitle, font=regular(22), fill=muted)
    return image

def save(image, name):
    path = output / f'{name}.png'
    # RGB PNG optimization only: no palette quantization, no UI text resampling.
    image.save(path, optimize=True, compress_level=9)
    if path.stat().st_size > 900_000:
        raise ValueError(f'{path}: {path.stat().st_size:,} bytes exceeds 900 KB')
    print(f'{path}: {image.width}x{image.height}, {path.stat().st_size:,} bytes')

save(capture('brick', 'split'), 'hero')
save(capture('brick', 'export-options'), 'export')

maps = canvas('Seven maps. One source photo.', 'Rock wall / generated in Texelith / individual map previews')
draw = ImageDraw.Draw(maps)
for index, (state, label) in enumerate([
    ('Albedo', 'Albedo'), ('Height', 'Height'), ('Normal', 'Normal'), ('AO', 'Ambient occlusion'),
    ('Roughness', 'Roughness'), ('Metallic', 'Metallic'), ('Edge', 'Edge / curvature'),
]):
    x, y = 24 + (index % 4) * 472, 126 + (index // 4) * 466
    draw.rounded_rectangle((x, y, x + 455, y + 449), radius=10, fill=panel)
    # Only texture pixels are resized for the contact sheet; labels are drawn at native size.
    tile = capture('rock', state).crop((396, 193, 1204, 1001))
    tile = tile.resize((360, 360), Image.Resampling.LANCZOS)
    maps.paste(tile, (x + 48, y + 24))
    draw.text((x + 24, y + 420), label, font=regular(20), fill=white)
x, y = 1440, 592
draw.text((x + 24, y + 70), 'PHOTO TO PBR', font=bold(28), fill=amber)
draw.text((x + 24, y + 125), 'Inspect each map.\nTune the result.\nExport for your engine.', font=regular(25), fill=white, spacing=15)
draw.text((x + 24, y + 325), 'Rock is dielectric:\nits metallic map stays black.', font=regular(21), fill=muted, spacing=8)
save(maps, 'maps')

materials = canvas('Four surfaces. The same light.', 'Preview ball / procedural sky / identical camera and exposure', 752)
draw = ImageDraw.Draw(materials)
for index, (key, label) in enumerate([
    ('brick', 'Red brick'), ('rock', 'Rock wall'), ('wood', 'Wood planks'), ('metal', 'Rusty metal'),
]):
    x = 24 + index * 472
    # Reduce only viewport pixels; all labels are drawn at native resolution.
    tile = capture(key, '3d').crop((572, 290, 1028, 940))
    tile = tile.resize((352, 502), Image.Resampling.LANCZOS)
    materials.paste(tile, (x + 52, 130))
    draw.text((x + 52, 674), label, font=bold(25), fill=white)
save(materials, 'materials')

seams = canvas('Make the repeat work.', 'Wood planks / actual albedo tiled 3 x 3 / default offset blend', 1040)
draw = ImageDraw.Draw(seams)
for index, (state, label, note) in enumerate([
    ('before', 'BEFORE / original edges', 'Visible vertical joins'),
    ('after', 'AFTER / Make Seamless', 'Blended tile boundaries'),
]):
    x = 164 + index * 944
    draw.text((x, 129), label, font=bold(25), fill=amber if index else white)
    # The GPU tiled viewport is left-aligned; the diagnostic presenter is centered.
    # Both are app-rendered captures. Keep the albedo unobscured and unadjusted.
    tile = capture('wood', f'tiled-{state}').crop((88, 193, 896, 1001))
    tile = tile.resize((600, 600), Image.Resampling.LANCZOS)
    seams.paste(tile, (x, 177))
    draw.text((x, 790), note, font=regular(22), fill=white)
    inset = capture('wood', f'seams-{state}').crop((396, 193, 1204, 1001))
    inset = inset.resize((112, 112), Image.Resampling.LANCZOS)
    seams.paste(inset, (x, 837))
    draw.rectangle((x - 1, 836, x + 112, 949), outline=muted)
    draw.text((x + 136, 850), 'Seam heatmap / diagnostic', font=bold(20), fill=white)
    draw.text((x + 136, 885), 'Warm colors = greater edge mismatch', font=regular(20), fill=muted)
draw.text((164, 987), 'Offset blending changes the pattern. Results depend on the source photo.', font=regular(22), fill=muted)
save(seams, 'seamless')

names = ['hero', 'maps', 'materials', 'seamless', 'export']
total = sum((output / f'{name}.png').stat().st_size for name in names)
if total > 5_000_000:
    raise ValueError(f'Gallery exceeds 5 MB: {total:,} bytes')
print(f'Gallery total: {total:,} bytes (lossless RGB PNG)')
'@ | python - $CaptureDirectory
if ($LASTEXITCODE -ne 0) { throw 'Gallery assembly or size validation failed' }
