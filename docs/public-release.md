```powershell
$release = Join-Path ([IO.Path]::GetTempPath()) 'texelith-public'
./scripts/export-public.ps1 -Out $release
Push-Location $release
cargo build --workspace --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all -- --check
git init
git config user.name 'Texelith'
git config user.email 'noreply@example.com'
git add .
git diff --cached --check
git commit -m 'Initial public release'
Pop-Location
```
