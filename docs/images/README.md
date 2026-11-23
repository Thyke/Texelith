# README gallery

The gallery uses Texelith's own `--ui-snapshot --readme-snapshots` mode at **1920×1080, scale 1.0**. Full screenshots retain every captured pixel; composite labels are drawn at native size. Texture and viewport crops are reduced for the composites; captured UI text is never resized. PNG encoding is lossless RGB: no palette quantization or resized UI text.

## Sources

Sample textures: Poly Haven (CC0). The original diffuse 2K JPGs are local screenshot inputs only and are not redistributed.

| Input file | Source | Author |
|---|---|---|
| `red_brick_03_diff_2k.jpg` | [Red Brick 03](https://polyhaven.com/a/red_brick_03) | Rob Tuytel |
| `rock_wall_08_diff_2k.jpg` | [Rock Wall 08](https://polyhaven.com/a/rock_wall_08) | Amal Kumar |
| `wood_planks_diff_2k.jpg` | [Wood Planks](https://polyhaven.com/a/wood_planks) | Amal Kumar |
| `rusty_metal_02_diff_2k.jpg` | [Rusty Metal 02](https://polyhaven.com/a/rusty_metal_02) | Rob Tuytel |

## Reproduce

On Windows, with the repository Rust toolchain and Python with Pillow available, place the four files in the git-ignored `target/readme-inputs/`, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/readme-images.ps1
```

The script builds the app, runs four sequential self-capturing sessions and assembles the five gallery images. No foreground screenshot tool is required. Raw captures and GPU validation reports stay under git-ignored `target-evidence/T-168/`. To assemble again from those captures:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/readme-images.ps1 -SkipCapture
```

All four materials use the same Preview ball, camera, exposure and procedural sky. The rusty metal selects neutral gray with metallic color `[0.4, 0.4, 0.4]`, radius `0.12` and threshold `0.45`, subtracting the empty painted mask, and normal strength `0.3`; other material controls use their defaults. The seam comparison shows unobscured wood albedo tiled 3×3 before and after the default offset blend, with small diagnostic heatmap insets. Both views come from the app: `tiled-before` / `tiled-after` capture albedo, and `seams-before` / `seams-after` capture the heatmap. No color or brightness adjustment is applied. Export uses the Unreal preset and the neutral example path `C:\Textures\Texelith`.

The assembly script checks every source capture is 1920×1080 and rejects any PNG over 900,000 bytes or a gallery total over 5,000,000 bytes. It does not fetch textures, create exports or publish files.
