param([switch]$SelfTest)
$ErrorActionPreference = 'Stop'
$versions = @{ rfd='0.18.0'; eframe='0.37.0'; egui='0.37.0'; 'egui-wgpu'='0.37.0'; wgpu='30.0.1'; image='0.25.10'; serde='1.0.229'; serde_json='1.0.154'; blake3='1.8.7'; 'crossbeam-channel'='0.5.17'; bytemuck='1.25.2'; pollster='1.0.1'; exr='1.74.3'; gltf='1.4.1'; tobj='4.0.5'; 'fluent-bundle'='0.16.0'; 'unic-langid'='0.9.6' }
$features = @{ eframe='clipboard,default_fonts,wayland,wgpu,x11'; image='hdr,jpeg,png'; gltf='utils'; serde='derive'; bytemuck='derive' }
$direct = @{ 'texelith-core'=@('serde','blake3','serde_json'); 'texelith-gpu'=@('wgpu','bytemuck','crossbeam-channel','serde','serde_json','pollster'); 'texelith-io'=@('image','serde_json','exr','gltf','tobj'); 'texelith-ui-egui'=@('rfd','eframe','egui','egui-wgpu','wgpu','serde','serde_json','crossbeam-channel','fluent-bundle','unic-langid') }
function Check($m, $manifests) {
    $errors = @()
    # TOML keys can be bare, basic quoted, or literal quoted, including dotted tables.
    $forbiddenKey = '(?:patch|replace|"(?:patch|replace)"|''(?:patch|replace)'')'
    foreach ($raw in $manifests) { if ($raw -match "(?m)^\s*(?:\[\s*$forbiddenKey\s*(?:\.|\])|$forbiddenKey\s*(?:\.|=))" -or $raw -match '(?m)\bgit\s*=') { $errors += 'Git dependencies and patch/replace sections are forbidden' } }
    foreach ($p in @($m.packages | Where-Object { $m.workspace_members -contains $_.id })) {
        foreach ($d in $p.dependencies) {
            if ($d.path) { continue }
            if (-not $versions.ContainsKey($d.name) -or $d.name -notin $direct[$p.name]) { $errors += "Unapproved direct dependency $($p.name): $($d.name)"; continue }
            $expectedFeatures = if ($features.ContainsKey($d.name)) { $features[$d.name] } else { '' }
            $actualFeatures = (@($d.features | Sort-Object) -join ',')
            $expectedDefault = $d.name -notin @('eframe','image','exr','gltf','tobj')
            if ($d.req -ne "=$($versions[$d.name])" -or $actualFeatures -ne $expectedFeatures -or $d.uses_default_features -ne $expectedDefault -or $d.rename -or $d.optional -or $d.target -or ($d.kind -and -not ($d.kind -eq 'dev' -and (($p.name -eq 'texelith-core' -and $d.name -eq 'serde_json') -or ($p.name -eq 'texelith-gpu' -and $d.name -eq 'pollster')))) -or ($d.name -eq 'pollster' -and $d.kind -ne 'dev') -or ($p.name -eq 'texelith-core' -and $d.name -eq 'serde_json' -and $d.kind -ne 'dev')) { $errors += "Dependency policy mismatch: $($p.name) / $($d.name)" }
        }
    }
    foreach ($p in $m.packages) {
        if ($p.source) {
            $allowedLicense = '^(MIT|Apache-2\.0|BSD-2-Clause|BSD-3-Clause|ISC|Zlib|Unicode-3\.0|BSL-1\.0)$'
            $legacyFont = $p.name -eq 'epaint_default_fonts' -and $p.version -eq '0.37.0'
            $clauses = @($p.license -split '\s+AND\s+')
            foreach ($clause in $clauses) {
                $choices = @($clause.Trim(' ', '(', ')') -split '(?:\s+OR\s+|\s*/\s*)')
                $allowed = @($choices | Where-Object { $_ -match $allowedLicense }).Count -gt 0
                if (-not $allowed -and -not ($legacyFont -and $clause -in @('OFL-1.1','Ubuntu-font-1.0'))) { $errors += "Unapproved license: $($p.name) $($p.version) $($p.license)" }
            }
        }
        if (-not $p.source -and $m.workspace_members -notcontains $p.id) { $errors += "Unexpected local package: $($p.name) $($p.manifest_path)" }
        if ($p.source -and $p.source -notlike 'registry+https://github.com/rust-lang/crates.io-index*') { $errors += "Unapproved source: $($p.name) $($p.source)" }
        if ($p.name -in @('egui','egui-wgpu','eframe') -and $p.version -ne '0.37.0') { $errors += "Mismatched egui family: $($p.name) $($p.version)" }
        if ($p.name -eq 'wgpu' -and $p.version -ne '30.0.1') { $errors += "Mismatched wgpu: $($p.version)" }
    }
    return ,$errors
}
$json = & cargo metadata --format-version 1 --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$m = $json | ConvertFrom-Json
$manifests = @((Get-Content (Join-Path $m.workspace_root 'Cargo.toml') -Raw)) + @($m.packages | Where-Object { $m.workspace_members -contains $_.id } | ForEach-Object { Get-Content $_.manifest_path -Raw })
if ($SelfTest) {
    if ((Check $m $manifests).Count) { throw 'Baseline dependency policy rejected' }
    foreach ($case in @('crate','version','feature','git','patch','quoted-patch','local-patch','egui','wgpu','license','font-version','new-feature')) {
        $copy = $json | ConvertFrom-Json
        $p = $copy.packages | Where-Object name -eq 'texelith-core'
        $d = $p.dependencies | Where-Object name -eq 'serde'
        $inputs = $manifests
        switch ($case) {
            crate { $d.name = 'unapproved' }
            version { $d.req = '=1.0.0' }
            feature { $d.features = @('derive','extra') }
            git { $d.source = 'git+https://example.invalid/repo' ; $inputs = @('dep = { git = "example" }') }
            patch { $inputs = @('[patch.crates-io]') }
            'quoted-patch' { $inputs = @(Get-Content (Join-Path $PSScriptRoot 'policy-fixtures/quoted-patch.toml') -Raw) }
            'local-patch' { $p = $copy.packages | Where-Object { $_.name -eq 'serde' } | Select-Object -First 1; $p.source = $null; $p.manifest_path = Join-Path $m.workspace_root 'vendor/serde/Cargo.toml' }
            license { ($copy.packages | Where-Object name -eq 'exr' | Select-Object -First 1).license = 'GPL-3.0-only' }
            'font-version' { ($copy.packages | Where-Object name -eq 'epaint_default_fonts' | Select-Object -First 1).version = '0.37.1' }
            'new-feature' { $p = $copy.packages | Where-Object name -eq 'texelith-io'; ($p.dependencies | Where-Object name -eq 'gltf').features = @('utils','import') }
            egui { ($copy.packages | Where-Object name -eq 'egui' | Select-Object -First 1).version = '0.1.0' }
            wgpu { ($copy.packages | Where-Object name -eq 'wgpu' | Select-Object -First 1).version = '0.1.0' }
        }
        if (-not (Check $copy $inputs).Count) { throw "Self-test did not reject $case" }
        Write-Host "PASS: $case rejected"
    }
    Write-Host 'PASS: baseline accepted'
} else {
    $errors = Check $m $manifests
    if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ -ErrorAction Continue }; exit 1 }
    Write-Host 'Dependency policy passed. Licenses are checked separately by cargo-deny.'
}
