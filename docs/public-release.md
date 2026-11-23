Export and verify the clean public tree before initializing its release history.
The CPU checks compile every example and test without private reference data.
`TEXELITH_CI_NO_GPU=1` skips hardware assertions; native window and GPU performance
checks remain separate hardware gates. Reports and captures default to
`target/evidence/<suite>`; use `--report` to select another location when supported.

```powershell
$release = Join-Path ([IO.Path]::GetTempPath()) 'texelith-public'
./scripts/export-public.ps1 -Out $release
$previousNoGpu = $env:TEXELITH_CI_NO_GPU
Push-Location $release
try {
    $env:TEXELITH_CI_NO_GPU = '1'
    cargo fmt --all -- --check
    if ($LASTEXITCODE) { throw 'Public format check failed' }
    cargo clippy --workspace --all-targets --locked -- -D warnings
    if ($LASTEXITCODE) { throw 'Public lint/all-target compilation failed' }
    cargo test --workspace --locked
    if ($LASTEXITCODE) { throw 'Public CPU tests failed' }
    cargo +1.95.0 check --workspace --all-targets --locked
    if ($LASTEXITCODE) { throw 'Public MSRV check failed' }
} finally {
    Pop-Location
    $env:TEXELITH_CI_NO_GPU = $previousNoGpu
}
```

After successful verification, initialize the public repository in that directory:

```powershell
Push-Location $release
git init
git config user.name 'Texelith'
git config user.email 'noreply@example.com'
git add .
git diff --cached --check
git commit -m 'Initial public release'
Pop-Location
```
