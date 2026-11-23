# Windows packaging

On Windows x64 with Rust and the Windows SDK installed, run:

```powershell
./scripts/package-windows.ps1 -Verify
```

The unsigned ZIP is written to `target-evidence/T-091/package/Texelith-0.1.0-windows-x64.zip` along with its SHA-256 hash. Extract it to any folder and run `Texelith.exe`, or run `install.ps1` for a per-user installation and Start Menu shortcut. The installer and uninstaller preserve user projects. Windows may show a trust warning for this unsigned build. The executable includes the statically linked MSVC runtime; the Windows SDK is needed to build it, not to run it.

`-Verify` exercises installation and removal in temporary paths; visual rendering and startup timing require a separate native-window check.
