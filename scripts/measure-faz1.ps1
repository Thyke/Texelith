param(
    [string]$AdapterName = 'RTX 5070 Ti',
    [ValidateSet('dx12','vulkan','metal')][string]$Backend = 'dx12',
    [ValidateRange(1,10)][int]$Runs = 3,
    [ValidateSet('canonical','unreal-v1','exr','unity-urp-v1','unity-hdrp-v1','godot-v1','blender-v1','gltf2-v1')]
    [string[]]$Profiles = @('canonical','unreal-v1','exr','unity-urp-v1','unity-hdrp-v1','godot-v1','blender-v1','gltf2-v1'),
    [string]$OutputDir = 'target-evidence/T-096/perf',
    [switch]$SelfTest,
    # Sandbox-safe diagnostic only: never reports the full acceptance gate passed.
    [switch]$ExportOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo

function Validate-Trial($Data, [int]$Code, [string]$Profile, [string]$Quality) {
    $reasons = @()
    if ($Code -ne 0) { $reasons += "process exit $Code" }
    if ($null -eq $Data) { return @($reasons + 'missing report') }
    if ($Data.passed -ne $true) { $reasons += 'app gate failed' }
    # Pre-admission errors have no measurement counters; do not invent GPU errors.
    if ($Data.error) { return @($reasons + [string]$Data.error) }
    if ($Data.profile -ne $Profile -or $Data.quality -ne $Quality) { $reasons += 'trial identity mismatch' }
    if ($Data.size -ne 4096) { $reasons += 'not 4K' }
    if ($Data.cache_invalidated -ne $true -or $Data.compute_submissions -le 0 -or $Data.cache_misses -le 0) { $reasons += 'no invalidation/recomputation evidence' }
    if ($null -eq $Data.metric -or $null -eq $Data.metric.request_to_disk_ns -or
        $Data.metric.request_to_disk_ns -le 0 -or $Data.metric.request_to_disk_ns -ge 2000000000) { $reasons += 'request_to_disk_ns must be positive and <2 s' }
    if ($null -eq $Data.validation_errors) { $reasons += 'missing GPU validation counter' }
    elseif ($Data.validation_errors -ne 0) { $reasons += 'GPU validation errors' }
    if (@($Data.files).Count -eq 0 -or @($Data.metric.outputs).Count -ne @($Data.files).Count) { $reasons += 'incomplete output evidence' }
    if ($Profile -in @('canonical','exr') -and @($Data.files).Count -ne 7) { $reasons += 'expected seven images' }
    return $reasons
}

if ($SelfTest) {
    $good = '{"passed":true,"profile":"canonical","quality":"default","size":4096,"cache_invalidated":true,"compute_submissions":7,"cache_misses":7,"validation_errors":0,"metric":{"request_to_disk_ns":1999999999,"outputs":[1,2,3,4,5,6,7]},"files":[1,2,3,4,5,6,7]}'
    if (@(Validate-Trial ($good | ConvertFrom-Json) 0 'canonical' 'default').Count -ne 0) { throw 'SelfTest: valid trial rejected' }
    foreach ($mutation in @('over','zero','cache','compute','misses','size','files','failure','identity')) {
        $data = $good | ConvertFrom-Json
        switch ($mutation) {
            'over' { $data.metric.request_to_disk_ns = 2000000000 }
            'zero' { $data.metric.request_to_disk_ns = 0 }
            'cache' { $data.cache_invalidated = $false }
            'compute' { $data.compute_submissions = 0 }
            'misses' { $data.cache_misses = 0 }
            'size' { $data.size = 1024 }
            'files' { $data.files = @(1) }
            'failure' { $data.passed = $false }
            'identity' { $data.profile = 'unreal-v1' }
        }
        if (@(Validate-Trial $data 0 'canonical' 'default').Count -eq 0) { throw "SelfTest: accepted $mutation" }
    }
    if (@(Validate-Trial ($good | ConvertFrom-Json) 1 'canonical' 'default').Count -eq 0 -or
        @(Validate-Trial $null 0 'canonical' 'default').Count -eq 0) { throw 'SelfTest: failure propagation broken' }
    $preflight = @(Validate-Trial ('{"passed":false,"error":"codec admission failed"}' | ConvertFrom-Json) 1 'canonical' 'default')
    if ($preflight -notcontains 'codec admission failed' -or $preflight -contains 'GPU validation errors') { throw 'SelfTest: missing counters confused with actual GPU errors' }
    & cargo test -p texelith-ui-egui --test mvp_t096 --locked
    if ($LASTEXITCODE -ne 0) { throw 'SelfTest: real metric/cache/publication behavioral contracts failed' }
    Write-Host 'PASS: boundaries, failure propagation, recomputation, immutable publication, script validator'
    exit 0
}

$suite = Join-Path ([IO.Path]::GetFullPath($OutputDir)) ('suite-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
[IO.Directory]::CreateDirectory($suite) | Out-Null
function Save-Json($Name, $Value) {
    $Value | ConvertTo-Json -Depth 60 | Set-Content -LiteralPath (Join-Path $suite $Name) -Encoding UTF8
}
function Logged-Cargo($Name, [string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    & cargo @Arguments *> (Join-Path $suite "$Name.log")
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { throw "$Name failed ($code); see $suite" }
}
try {
    if (@($Profiles | Select-Object -Unique).Count -ne $Profiles.Count) { throw 'Duplicate profiles would overwrite trial evidence' }
    Logged-Cargo 'build-release' @('build','-p','texelith-ui-egui','--release','--locked')
    $sources = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml') + @(& rg --files crates scripts .cargo)
    $hashes = @($sources | Sort-Object -Unique | ForEach-Object {
        @{ path=$_; sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
    })
    Save-Json 'source-manifest.json' $hashes
    $environment = [ordered]@{
        recorded_at=(Get-Date).ToString('o'); revision=(& git rev-parse HEAD); dirty_paths=@(& git status --short)
        rust=(& rustc --version); backend=$Backend; requested_adapter=$AdapterName; os=[Environment]::OSVersion.VersionString
        destination=$suite; disk=$null; disk_query_error=$null; disk_verified_ssd=$false
        other_gpu_activity='not isolated by script; orchestrator must provide exclusive hardware slot'
        memory_scope='Parent-sampled process lifetime peak working set, including fixture/decode/warmup/verification; GPU pool high-water reported separately'
    }
    try {
        $driveLetter = [IO.Path]::GetPathRoot($suite).Substring(0,1)
        $disk = Get-Partition -DriveLetter $driveLetter -ErrorAction Stop | Get-Disk -ErrorAction Stop
        $physical = @(Get-PhysicalDisk -ErrorAction Stop | Where-Object { $_.DeviceId -eq [string]$disk.Number })
        $environment.disk = @{ number=$disk.Number; name=$disk.FriendlyName; bus=[string]$disk.BusType; physical=@($physical | Select-Object FriendlyName,MediaType,BusType) }
        $environment.disk_verified_ssd = @($physical | Where-Object { [string]$_.MediaType -eq 'SSD' }).Count -gt 0
    } catch { $environment.disk_query_error = $_.ToString() }
    Save-Json 'environment.json' $environment
    $faz0 = 'deferred to orchestrator'
    if (-not $ExportOnly) {
        # Invoke the unchanged original gate, preserving all original thresholds.
        $ErrorActionPreference = 'Continue'
        & powershell -NoProfile -File (Join-Path $PSScriptRoot 'measure-faz0.ps1') -AdapterName $AdapterName -Backend $Backend -Runs $Runs -OutputDir (Join-Path $suite 'faz0') *> (Join-Path $suite 'faz0.log')
        $faz0Code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        $faz0 = if ($faz0Code -eq 0) { 'PASS' } else { 'FAIL' }
    }
    $results = @()
    foreach ($quality in @('default','maximum')) {
        foreach ($profile in $Profiles) {
            for ($run = 1; $run -le $Runs; $run++) {
                $name = "$profile-$quality-$run"
                $report = Join-Path $suite "$name.json"
                if ($report.Contains('"') -or $AdapterName.Contains('"')) { throw 'Quotes in native arguments unsupported' }
                $info = New-Object Diagnostics.ProcessStartInfo
                $info.FileName = Join-Path $repo 'target/release/texelith-ui-egui.exe'
                $info.WorkingDirectory = $repo
                $info.Arguments = '--export-perf --size 4096 --profile ' + $profile + ' --quality ' + $quality + ' --backend ' + $Backend + ' --adapter-name "' + $AdapterName + '" --report "' + $report + '"'
                $info.UseShellExecute = $false
                $info.CreateNoWindow = $true
                $info.RedirectStandardOutput = $true
                $info.RedirectStandardError = $true
                $process = New-Object Diagnostics.Process
                $process.StartInfo = $info
                [void]$process.Start()
                $stdout = $process.StandardOutput.ReadToEndAsync()
                $stderr = $process.StandardError.ReadToEndAsync()
                $deadline = [Diagnostics.Stopwatch]::StartNew()
                $peak = 0L
                $timedOut = $false
                while (-not $process.WaitForExit(100)) {
                    $process.Refresh()
                    $peak = [math]::Max($peak, $process.PeakWorkingSet64)
                    if ($deadline.Elapsed.TotalSeconds -gt 420) {
                        $timedOut = $true
                        $process.Kill()
                        if (-not $process.WaitForExit(5000)) { throw "$name could not be terminated" }
                        break
                    }
                }
                $process.Refresh()
                $peak = [math]::Max($peak, $process.PeakWorkingSet64)
                if (-not $stdout.Wait(5000) -or -not $stderr.Wait(5000)) { throw "$name output pipes stuck" }
                [IO.File]::WriteAllText((Join-Path $suite "$name.stdout.log"), $stdout.Result)
                [IO.File]::WriteAllText((Join-Path $suite "$name.stderr.log"), $stderr.Result)
                $data = $null
                $parseError = $null
                if (Test-Path -LiteralPath $report) {
                    try { $data = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json } catch { $parseError = $_.ToString() }
                }
                $reasons = @(Validate-Trial $data $process.ExitCode $profile $quality)
                if ($timedOut) { $reasons += 'timeout' }
                if ($parseError) { $reasons += $parseError }
                $entry = [ordered]@{ profile=$profile; quality=$quality; run=$run; report=$report; exit_code=$process.ExitCode
                    passed=($reasons.Count -eq 0); failures=$reasons; peak_working_set_bytes=$peak; data=$data }
                $results += $entry
                Save-Json 'summary.json' @{ passed=$false; native_faz0=$faz0; environment=$environment; trials=$results }
                $ms = if ($null -ne $data.metric) { [math]::Round($data.metric.request_to_disk_ns / 1e6, 3) } else { 'not measured' }
                Write-Host "$name request->disk=$ms ms PASS=$($entry.passed) failures=$($reasons -join '; ')"
                $process.Dispose()
            }
        }
    }
    $numeric = $Runs -eq 3 -and $results.Count -eq 48 -and @($results | Where-Object { -not $_.passed }).Count -eq 0
    $passed = $numeric -and $faz0 -eq 'PASS' -and $environment.disk_verified_ssd
    Save-Json 'summary.json' @{ passed=$passed; numeric_passed=$numeric; native_faz0=$faz0; environment=$environment; trials=$results }
    Write-Host "Evidence: $suite"
    if ($passed) { exit 0 } else { exit 1 }
} catch {
    Save-Json 'script-failure.json' @{ passed=$false; error=$_.ToString() }
    Write-Error $_ -ErrorAction Continue
    exit 1
}
