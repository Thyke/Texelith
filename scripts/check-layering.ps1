param([switch]$SelfTest, [switch]$ValidateWorkflow)
$ErrorActionPreference = 'Stop'
$allowed = @{ 'texelith-core'=@(); 'texelith-gpu'=@('texelith-core'); 'texelith-io'=@('texelith-core'); 'texelith-ui-egui'=@('texelith-core','texelith-gpu','texelith-io') }
function PackagePath($root, $name) { return [IO.Path]::GetFullPath((Join-Path $root "crates/$name")).TrimEnd('\','/').ToLowerInvariant() }
function CanonicalPath($path) { return [IO.Path]::GetFullPath($path).TrimEnd('\','/').ToLowerInvariant() }
function Check($m) {
    $errors = @()
    $members = @($m.packages | Where-Object { $m.workspace_members -contains $_.id })
    foreach ($p in $members) {
        if (-not $allowed.ContainsKey($p.name)) { $errors += "Unexpected workspace package: $($p.name)"; continue }
        if ((CanonicalPath (Split-Path $p.manifest_path)) -ne (PackagePath $m.workspace_root $p.name)) { $errors += "Unexpected workspace package path: $($p.name)" }
        foreach ($d in $p.dependencies) {
            if ($d.path -or $d.name -like 'texelith-*') {
                if ($d.name -notin $allowed[$p.name] -or $d.rename -or $d.kind -or $d.target -or $d.optional) { $errors += "Forbidden local edge: $($p.name) -> $($d.name)" }
                if (-not $d.path -or -not $allowed.ContainsKey($d.name) -or (CanonicalPath $d.path) -ne (PackagePath $m.workspace_root $d.name)) { $errors += "Unexpected local dependency path: $($p.name) -> $($d.name)" }
            }
        }
    }
    foreach ($name in $allowed.Keys) { if ($name -notin $members.name) { $errors += "Missing workspace package: $name" } }
    return ,$errors
}
function CheckWorkflow([string]$workflow) {
    $errors = @()
    $required = @(
        'name: CI', '  push:', '  pull_request:', 'shell: pwsh',
        'TEXELITH_CI_NO_GPU: ''1''', 'cancel-in-progress: true',
        'cargo fmt --all -- --check',
        'cargo clippy --workspace --all-targets --locked -- -D warnings',
        'cargo test --workspace --locked -- --nocapture',
        'check-layering.ps1 -SelfTest', 'check-layering.ps1 -ValidateWorkflow',
        'check-dependencies.ps1 -SelfTest', 'cargo deny check licenses bans sources',
        'cargo +1.95.0 check --workspace --all-targets --locked',
        'rustup toolchain install 1.95.0 --profile minimal',
        'rustup toolchain install 1.99.1 --profile minimal --component rustfmt --component clippy --target wasm32-unknown-unknown',
        'cargo check -p texelith-core --target wasm32-unknown-unknown --locked',
        'install-linux-ci-deps.ps1',
        'actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683',
        'actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9'
    )
    foreach ($item in $required) { if (-not $workflow.Contains($item)) { $errors += "Missing workflow contract: $item" } }
    $sections = @{}
    foreach ($job in @('native','policy','portability')) {
        $match = [regex]::Match($workflow, "(?ms)^  ${job}:\s*\r?\n(.*?)(?=^  [a-z][a-z-]*:\s*\r?$|\z)")
        if (-not $match.Success) { $errors += "Missing job: $job" } else { $sections[$job] = $match.Groups[1].Value }
    }
    $native = $sections['native']
    if ($native -notmatch '(?m)^    runs-on: \$\{\{ matrix\.os \}\}\s*$') { $errors += 'Native runner must bind to matrix.os' }
    if ($native -notmatch '(?m)^        os: \[windows-latest, ubuntu-latest, macos-latest\]\s*$') { $errors += 'All native platforms must run on every push/PR' }
    foreach ($gate in @('cargo fmt --all -- --check','cargo clippy --workspace --all-targets --locked -- -D warnings','cargo test --workspace --locked -- --nocapture')) {
        if (-not $native.Contains($gate)) { $errors += "Missing native gate: $gate" }
    }
    foreach ($job in $sections.Keys) {
        if ($sections[$job] -match '(?m)^    if:' -or $sections[$job] -match 'continue-on-error:' -or $sections[$job] -match '--skip ') { $errors += "Disabled or masked gate: $job" }
        foreach ($condition in [regex]::Matches($sections[$job], '(?m)^ +if:.*$')) {
            if ($condition.Value.Trim() -notin @("if: runner.os == 'Linux'", 'if: always()')) { $errors += "Unexpected condition in $job" }
        }
    }
    if ($workflow -match 'measure-faz|--offline|github.event_name|fromJSON') { $errors += 'Hosted gates must not require a window, offline cache, or event-dependent matrix' }
    foreach ($action in [regex]::Matches($workflow, '(?m)^ +(?:- )?uses: (\S+)')) {
        if ($action.Groups[1].Value -notmatch '@[0-9a-f]{40}$') { $errors += 'Unpinned action' }
    }
    return ,$errors
}
if ($ValidateWorkflow) {
    $workflow = Get-Content (Join-Path $PSScriptRoot '../.github/workflows/ci.yml') -Raw
    $errors = CheckWorkflow $workflow
    if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ -ErrorAction Continue }; exit 1 }
    foreach ($marker in @('windows-latest','macos-latest','ubuntu-latest','cargo deny check licenses bans sources','TEXELITH_CI_NO_GPU: ''1''')) {
        if (-not (CheckWorkflow $workflow.Replace($marker, 'REMOVED')).Count) { throw "Workflow validator missed removal of $marker" }
    }
    $mutations = @{
        'fixed runner' = $workflow.Replace('runs-on: ${{ matrix.os }}', 'runs-on: windows-latest')
        'disabled native job' = $workflow.Replace('  native:', "  native:`n    if: false")
        'disabled format gate' = $workflow.Replace('      - name: Format', "      - name: Format`n        if: false")
        'masked test failure' = $workflow.Replace('      - name: Tests', "      - continue-on-error: true`n      - name: Tests")
    }
    foreach ($name in $mutations.Keys) {
        if (-not (CheckWorkflow $mutations[$name]).Count) { throw "Workflow validator missed mutation: $name" }
    }
    Write-Host 'Workflow contract markers passed (static validation only).'
    exit 0
}
$json = & cargo metadata --format-version 1 --locked --no-deps
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$m = $json | ConvertFrom-Json
if ($SelfTest) {
    if ((Check $m).Count) { throw 'Baseline graph rejected' }
    $cases = @('upward','alias','target','build','dev','optional','package','impostor')
    foreach ($case in $cases) {
        $copy = $json | ConvertFrom-Json
        $core = $copy.packages | Where-Object name -eq 'texelith-core'
        $gpu = $copy.packages | Where-Object name -eq 'texelith-gpu'
        $edge = $gpu.dependencies | Where-Object name -eq 'texelith-core'
        switch ($case) {
            upward { $edge.name = 'texelith-ui-egui' }
            alias { $edge.rename = 'hidden' }
            target { $edge.target = 'cfg(windows)' }
            build { $edge.kind = 'build' }
            dev { $edge.kind = 'dev' }
            optional { $edge.optional = $true }
            package { $core.name = 'texelith-unreviewed' }
            impostor { $edge.path = Join-Path $m.workspace_root (Get-Content (Join-Path $PSScriptRoot 'policy-fixtures/impostor-path.txt') -Raw).Trim() }
        }
        if (-not (Check $copy).Count) { throw "Self-test did not reject $case" }
        Write-Host "PASS: $case rejected"
    }
    Write-Host 'PASS: ADR graph accepted'
} else {
    $errors = Check $m
    if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ -ErrorAction Continue }; exit 1 }
    Write-Host 'Layering policy passed.'
}
