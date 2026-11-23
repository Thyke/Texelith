param(
    [string]$AdapterName = 'RTX 5070 Ti',
    [ValidateSet('dx12','vulkan','metal')][string]$Backend = 'dx12',
    [ValidateRange(1,10)][int]$Runs = 3,
    [string]$OutputDir = 'docs/evidence/T-020'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$suite = Join-Path ([IO.Path]::GetFullPath($OutputDir)) "suite-$stamp"
[IO.Directory]::CreateDirectory($suite) | Out-Null
function Save-Json($Name, $Value) {
    $Value | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath (Join-Path $suite $Name) -Encoding UTF8
}
function Ms($Value) { if ($null -eq $Value) { 'not measured' } else { [math]::Round($Value / 1e6, 6) } }
function Invoke-Logged($Name, [string[]]$Arguments) {
    # Windows PowerShell 5 treats native stderr as ErrorRecord even on success.
    $ErrorActionPreference = 'Continue'
    & cargo @Arguments *> (Join-Path $suite "$Name.log")
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { throw "$Name failed ($code); see $suite" }
}
try {
    Invoke-Logged 'fixture' @('run','-p','texelith-io','--example','make_fixture','--release','--locked','--','--size','4096','--output','target/faz0-input.png')
    Invoke-Logged 'build' @('build','-p','texelith-ui-egui','--release','--locked')
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class Faz0Environment {
    [StructLayout(LayoutKind.Explicit, Size=220, CharSet=CharSet.Unicode)]
    public struct Mode {
        [FieldOffset(68)] public short size;
        [FieldOffset(172)] public int width;
        [FieldOffset(176)] public int height;
        [FieldOffset(184)] public int refresh;
    }
    [StructLayout(LayoutKind.Sequential)] public struct Power {
        public byte ac, flag, percent, reserved;
        public uint remaining, total;
    }
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern bool EnumDisplaySettings(string device, int mode, ref Mode value);
    [DllImport("kernel32.dll")] public static extern bool GetSystemPowerStatus(out Power value);
    [DllImport("user32.dll")] public static extern IntPtr GetDC(IntPtr window);
    [DllImport("user32.dll")] public static extern int ReleaseDC(IntPtr window, IntPtr dc);
    [DllImport("gdi32.dll")] public static extern int GetDeviceCaps(IntPtr dc, int index);
}
'@
    $mode = New-Object Faz0Environment+Mode
    $mode.size = 220
    $displayKnown = [Faz0Environment]::EnumDisplaySettings($null, -1, [ref]$mode)
    $refreshMethod = 'EnumDisplaySettings current primary mode'
    if (-not $displayKnown) {
        $dc = [Faz0Environment]::GetDC([IntPtr]::Zero)
        if ($dc -ne [IntPtr]::Zero) {
            try {
                $mode.refresh = [Faz0Environment]::GetDeviceCaps($dc, 116)
                $mode.width = [Faz0Environment]::GetDeviceCaps($dc, 8)
                $mode.height = [Faz0Environment]::GetDeviceCaps($dc, 10)
                $displayKnown = $mode.refresh -gt 1
                $refreshMethod = 'GetDeviceCaps VREFRESH fallback (0/1 means unknown)'
            } finally { [void][Faz0Environment]::ReleaseDC([IntPtr]::Zero, $dc) }
        }
    }
    $power = New-Object Faz0Environment+Power
    $powerKnown = [Faz0Environment]::GetSystemPowerStatus([ref]$power)
    $environment = [ordered]@{
        recorded_at = (Get-Date).ToString('o')
        worktree = $repo
        revision = (& git rev-parse HEAD)
        dirty_paths = @(& git status --short)
        rust = (& rustc --version)
        cargo = (& cargo --version)
        os = [Environment]::OSVersion.VersionString
        ac_connected = $(if ($powerKnown -and $power.ac -ne 255) { $power.ac -eq 1 } else { $null })
        primary_monitor_refresh_hz = $(if ($displayKnown) { $mode.refresh } else { $null })
        primary_monitor_size = @($mode.width, $mode.height)
        monitor_query = $refreshMethod
        monitor_scope = 'Primary display mode at launch; app uses default centered primary window. Physical scanout unmeasured.'
        present_mode = 'Fifo'
        target_hz = 60
        backend = $Backend
        requested_adapter = $AdapterName
        profile = 'release'
        debugger = 'launched directly by Process.Start; no debugger attached by this script'
        cold_start = 'QPC sampled immediately before Process.Start; first usable texture draw GPU completion reads same system counter; excludes cargo build and fixture creation, includes process launch overhead'
        cold_cache = 'new process each run; OS file and driver caches not flushed'
        other_gpu_activity = 'not isolated'
    }
    Save-Json 'environment.json' $environment
    $identity = @(Get-ChildItem crates -Recurse -File | Where-Object { $_.Extension -in '.rs','.wgsl','.toml' } | ForEach-Object {
        @{ path = $_.FullName.Substring($repo.Length + 1); sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
    })
    $identity += @{ path='Cargo.lock'; sha256=(Get-FileHash Cargo.lock -Algorithm SHA256).Hash }
    $identity += @{ path='target/faz0-input.png'; sha256=(Get-FileHash target/faz0-input.png -Algorithm SHA256).Hash }
    $identity += @{ path='scripts/measure-faz0.ps1'; sha256=(Get-FileHash $PSCommandPath -Algorithm SHA256).Hash }
    Save-Json 'source-manifest.json' $identity
    $results = @()
    for ($index = 1; $index -le $Runs; $index++) {
        $report = Join-Path $suite "run-$index.json"
        $info = New-Object Diagnostics.ProcessStartInfo
        $info.FileName = Join-Path $repo 'target/release/texelith-ui-egui.exe'
        $info.WorkingDirectory = $repo
        if ($AdapterName.Contains('"') -or $report.Contains('"')) { throw 'Quotes in process arguments are unsupported' }
        $info.Arguments = '--perf-scenario normal-drag --size 4096 --duration-seconds 10 --backend ' + $Backend + ' --adapter-name "' + $AdapterName + '" --input target/faz0-input.png --report "' + $report + '"'
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = New-Object Diagnostics.Process
        $process.StartInfo = $info
        $info.EnvironmentVariables['TEXELITH_PERF_LAUNCH_QPC'] = [Diagnostics.Stopwatch]::GetTimestamp().ToString([Globalization.CultureInfo]::InvariantCulture)
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        # Independent parent backstop: app's scenario deadline is duration+90 s,
        # followed by a 15 s shutdown deadline. Never wait forever after Kill.
        $finished = $process.WaitForExit(120000)
        if (-not $finished) {
            $process.Kill()
            if (-not $process.WaitForExit(5000)) { throw "Run $index could not be terminated within 5 seconds" }
        }
        if (-not $stdout.Wait(5000) -or -not $stderr.Wait(5000)) { throw "Run $index output pipes did not close" }
        [IO.File]::WriteAllText((Join-Path $suite "run-$index.stdout.log"), $stdout.Result)
        [IO.File]::WriteAllText((Join-Path $suite "run-$index.stderr.log"), $stderr.Result)
        $data = $null
        if (Test-Path -LiteralPath $report) { $data = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json }
        $lifecyclePath = [IO.Path]::ChangeExtension($report, 'lifecycle.json')
        $lifecycle = $null
        if (Test-Path -LiteralPath $lifecyclePath) { $lifecycle = Get-Content -LiteralPath $lifecyclePath -Raw | ConvertFrom-Json }
        $reasons = @()
        if ($null -eq $lifecycle) { $reasons += 'independent window/presentation observation missing' }
        else { $reasons += @($lifecycle.environment_reasons) }
        if ($null -ne $data -and $data.status -eq 'ENVIRONMENT_INVALID') { $reasons += @($data.environment_reasons) }
        $reasons = @($reasons | Sort-Object -Unique)
        $invalid = $reasons.Count -gt 0
        $numericStatus = if ($null -ne $data -and $null -ne $data.gate) {
            if (@($data.gate.violations).Count -eq 0) { 'PASS' } else { 'FAIL' }
        } else { 'NotMeasured' }
        $runPassed = $finished -and $process.ExitCode -eq 0 -and $null -ne $data -and $data.passed -and -not $invalid -and
            $null -ne $lifecycle.lifecycle.phases_ns.shutdown_completed -and -not $lifecycle.lifecycle.forced_exit
        $entry = [ordered]@{ run=$index; exit_code=$process.ExitCode; timed_out=(-not $finished); report=$report;
            slider_status=$(if ($invalid) { 'ENVIRONMENT_INVALID' } elseif ($runPassed) { 'PASS' } else { 'FAIL' });
            slider_numeric_status=$numericStatus; environment_reasons=$reasons; lifecycle_report=$lifecyclePath;
            lifecycle=$(if ($null -ne $lifecycle) { $lifecycle.lifecycle } else { $null });
            cold_start_status=$(if ($null -ne $data) { $data.gate.cold_start } else { 'NotMeasured' });
            cold_start_ns=$(if ($null -ne $data) { $data.trace.cold_start_ns } else { $null });
            gate=$(if ($null -ne $data) { $data.gate } else { $null });
            counters=$(if ($null -ne $data) { $data.trace.counters } else { $null }) }
        $results += $entry
        Save-Json 'summary.json' @{ expected_runs=$Runs; finished_runs=$results.Count; runs=$results; passed=$false }
        Write-Host "Run $index slider=$($entry.slider_status) numeric=$numericStatus cold_start=$($entry.cold_start_status) exit=$($entry.exit_code) reasons=$($reasons -join '; ')"
        $process.Dispose()
    }
    $passed = $Runs -eq 3 -and $results.Count -eq $Runs -and @($results | Where-Object { $_.slider_status -ne 'PASS' }).Count -eq 0
    $environmentMeasured = $environment.ac_connected -eq $true -and $displayKnown -and $mode.refresh -ge 60
    Save-Json 'summary.json' @{ expected_runs=$Runs; finished_runs=$results.Count; runs=$results; passed=($passed -and $environmentMeasured);
        slider_numeric_passed=($Runs -eq 3 -and @($results | Where-Object { $_.slider_numeric_status -ne 'PASS' }).Count -eq 0);
        valid_runs=@($results | Where-Object { $_.slider_status -ne 'ENVIRONMENT_INVALID' }).Count;
        environment_status=$(if (@($results | Where-Object { $_.slider_status -eq 'ENVIRONMENT_INVALID' }).Count -gt 0) { 'ENVIRONMENT_INVALID' } elseif ($environmentMeasured) { 'Measured' } else { 'NotMeasured' });
        cold_start_separate=$true; environment=$environment }
    $lines = @('# Faz 0 three-run measurement', '', "Suite: $suite", '', '| Run | Slider | Cold start | CPU p95 ms | Frame p95 / p99 / max ms | Preview p95 / gap ms | Frames / fresh | Release ms | Normal GPU / completion p95 ms | Peak MiB |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    foreach ($r in $results) {
        $m = $r.gate.metrics
        $peak = if ($null -eq $r.counters) { 'not measured' } else { $r.counters.peak_app_bytes/1MB }
        $lines += "| $($r.run) | $($r.slider_status) | $($r.cold_start_status) | $(Ms $m.ui_update_p95_ns) | $(Ms $m.frame_interval_p95_ns) / $(Ms $m.frame_interval_p99_ns) / $(Ms $m.frame_interval_max_ns) | $(Ms $m.proxy_age_p95_ns) / $(Ms $m.fresh_gap_max_ns) | $($m.drag_frames) / $($m.fresh_publications) | $(Ms $m.release_to_visible_ns) | $(Ms $m.normal_p95_ns) / $(Ms $m.normal_completion_p95_ns) | $peak |"
    }
    $lines += ''
    foreach ($r in $results) { $lines += "Run $($r.run) numeric=$($r.slider_numeric_status), violations: $($r.gate.violations -join ', '); environment: $($r.environment_reasons -join '; '). Lifecycle: $($r.lifecycle_report)" }
    $refreshLabel = if ($displayKnown) { "$($mode.refresh) Hz" } else { 'not measured' }
    $lines += "`nAC: $($environment.ac_connected); primary refresh: $refreshLabel; target: 60 Hz; Fifo. Physical scanout and manual checks not measured. All original reports retained; no replacement/averaging."
    $lines | Set-Content -LiteralPath (Join-Path $suite 'summary.md') -Encoding UTF8
    Write-Host "Evidence: $suite"
    if ($passed -and $environmentMeasured) { exit 0 } else { exit 1 }
} catch {
    Save-Json 'script-failure.json' @{ error=$_.ToString(); passed=$false }
    Write-Error $_ -ErrorAction Continue
    exit 1
}
