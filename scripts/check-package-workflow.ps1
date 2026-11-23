param([switch]$SelfTest, [string]$EvidenceRoot)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$workflowPath = Join-Path $repo '.github/workflows/package.yml'
$platforms = @{
    windows = @{ runner = 'windows-2022'; os = 'windows'; architecture = 'x86_64'; artifact = 'Texelith-0.1.0-windows-x64.zip' }
    macos = @{ runner = 'macos-latest'; os = 'macos'; architecture = 'aarch64'; artifact = 'Texelith-macos-arm64.dmg' }
    linux = @{ runner = 'ubuntu-22.04'; os = 'linux'; architecture = 'x86_64'; artifact = 'texelith_1.2.3_amd64.deb' }
}
function Assert($condition, $message) { if (-not $condition) { throw $message } }
function RemoveYamlComment([string]$line) {
    $quote = [char]0
    for ($i = 0; $i -lt $line.Length; $i++) {
        $c = $line[$i]
        if ($quote -eq [char]0) {
            if ($c -eq "'" -or $c -eq '"') { $quote = $c }
            elseif ($c -eq '#' -and ($i -eq 0 -or [char]::IsWhiteSpace($line[$i - 1]))) { return $line.Substring(0, $i).TrimEnd() }
        } elseif ($c -eq $quote) {
            if ($quote -eq "'" -and $i + 1 -lt $line.Length -and $line[$i + 1] -eq "'") { $i++ }
            else { $quote = [char]0 }
        } elseif ($quote -eq '"' -and $c -eq '\') { $i++ }
    }
    Assert ($quote -eq [char]0) 'Unterminated YAML quote'
    return $line.TrimEnd()
}
function ReadYamlNode($state, [int]$indent) {
    $sequence = $state.tokens[$state.index].item
    $map = @{}; $items = @()
    while ($state.index -lt $state.tokens.Count -and $state.tokens[$state.index].indent -eq $indent) {
        $token = $state.tokens[$state.index]; $state.index++
        Assert ($token.item -eq $sequence) 'Mixed YAML mapping and sequence'
        if ($sequence) {
            Assert ($state.index -lt $state.tokens.Count -and $state.tokens[$state.index].indent -gt $indent) 'Empty YAML sequence item'
            $items += ,(ReadYamlNode $state $state.tokens[$state.index].indent)
        } else {
            Assert (-not $map.ContainsKey($token.key)) "Duplicate YAML key: $($token.key)"
            $value = $token.value
            if ($state.index -lt $state.tokens.Count -and $state.tokens[$state.index].indent -gt $indent) {
                Assert ($null -eq $value) "Unexpected YAML children: $($token.key)"
                $value = ReadYamlNode $state $state.tokens[$state.index].indent
            }
            $map[$token.key] = $value
        }
    }
    if ($sequence) { return ,$items }
    return ,$map
}
function ReadWorkflowYaml([string]$yaml) {
    # Dependency-free, fail-closed subset: indentation maps, sequences of maps,
    # plain/quoted scalars and literal blocks. No aliases, tags or folded blocks.
    $lines = $yaml -split '\r?\n'; $tokens = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        Assert (-not $lines[$i].Contains("`t")) 'YAML tabs are unsupported'
        $line = RemoveYamlComment $lines[$i]
        if (-not $line.Trim()) { continue }
        $indent = $line.Length - $line.TrimStart().Length
        $text = $line.TrimStart()
        if ($text.StartsWith('- ')) {
            $tokens += @{ indent = $indent; item = $true }
            $indent += 2; $text = $text.Substring(2)
        }
        Assert ($text -cmatch '^([A-Za-z_][A-Za-z0-9_-]*):(?:\s+(.*))?$') "Unsupported YAML line: $line"
        $key = $Matches[1]; $value = $Matches[2]
        if ($value -eq '|') {
            $block = @(); $blockIndent = $null
            while ($i + 1 -lt $lines.Count) {
                $next = $lines[$i + 1]
                if ($next.Trim()) {
                    $nextIndent = $next.Length - $next.TrimStart().Length
                    if ($nextIndent -le $indent) { break }
                    if ($null -eq $blockIndent) { $blockIndent = $nextIndent }
                    Assert ($nextIndent -ge $blockIndent -and -not $next.Contains("`t")) 'Invalid YAML literal indentation'
                    $block += $next.Substring($blockIndent)
                } else { $block += '' }
                $i++
            }
            $value = ($block -join "`n") + "`n"
        } elseif ($value) {
            Assert ($value -notmatch '^[&*!>|{]' -and ($value -notmatch '^\[' -or $value -ceq "['v*']")) "Unsupported YAML scalar: $value"
            if ($value.StartsWith("'")) { $value = $value.Substring(1, $value.Length - 2).Replace("''", "'") }
            elseif ($value.StartsWith('"')) { $value = $value | ConvertFrom-Json }
        } else { $value = $null }
        $tokens += @{ indent = $indent; item = $false; key = $key; value = $value }
    }
    Assert ($tokens.Count -gt 0 -and $tokens[0].indent -eq 0) 'Missing YAML root'
    $state = @{ tokens = $tokens; index = 0 }
    $root = ReadYamlNode $state 0
    Assert ($state.index -eq $tokens.Count) 'Invalid YAML indentation'
    return $root
}
function AssertKeys($map, [string[]]$keys, [string]$label) {
    Assert ($map -is [System.Collections.IDictionary]) "Missing mapping: $label"
    Assert (@(Compare-Object @($map.Keys) $keys).Count -eq 0) "Unexpected keys in $label"
}
function ReadRunScript([string]$code) {
    $tokens = $null; $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseInput($code, [ref]$tokens, [ref]$parseErrors)
    Assert ($parseErrors.Count -eq 0) 'Invalid PowerShell run block'
    return $ast
}
function ScriptSignature([string]$code) {
    $tokens = $null; $parseErrors = $null
    $null = [System.Management.Automation.Language.Parser]::ParseInput($code, [ref]$tokens, [ref]$parseErrors)
    Assert ($parseErrors.Count -eq 0) 'Invalid PowerShell statement'
    return (($tokens | Where-Object { $_.Kind -notin @('Comment','NewLine','LineContinuation','EndOfInput') } | ForEach-Object { "$($_.Kind):$($_.Text)" }) -join "`n")
}
function RequireStatement($statements, [string]$expected, [string]$label) {
    $signature = ScriptSignature $expected
    $found = @($statements | Where-Object { (ScriptSignature $_.Extent.Text) -ceq $signature })
    Assert ($found.Count -eq 1) "Missing executable statement: $label"
}
function ValidateWorkflow([string]$yaml) {
    try {
        $w = ReadWorkflowYaml $yaml
        AssertKeys $w @('name','on','permissions','defaults','jobs') 'workflow'
        AssertKeys $w.on @('workflow_dispatch','push') 'triggers'
        Assert ($null -eq $w.on.workflow_dispatch) 'Unexpected dispatch configuration'
        AssertKeys $w.on.push @('tags') 'push trigger'
        Assert ($w.on.push.tags -ceq "['v*']") 'Missing version tag trigger'
        Assert ($w.permissions.contents -eq 'read' -and $w.defaults.run.shell -eq 'pwsh') 'Expected read permissions and pwsh'
        AssertKeys $w.jobs @('package') 'jobs'
        $job = $w.jobs.package
        AssertKeys $job @('strategy','runs-on','steps') 'package job'
        AssertKeys $job.strategy @('fail-fast','matrix') 'strategy'
        AssertKeys $job.strategy.matrix @('include') 'matrix'
        Assert ($job.strategy.'fail-fast' -eq 'false' -and $job.'runs-on' -ceq '${{ matrix.os }}') 'Expected independent native runners'
        $matrix = @($job.strategy.matrix.include); $seen = @{}
        Assert ($matrix.Count -eq 3) 'Expected exactly three matrix entries'
        foreach ($entry in $matrix) {
            AssertKeys $entry @('platform','os') 'matrix entry'
            Assert ($platforms.ContainsKey($entry.platform) -and -not $seen.ContainsKey($entry.platform)) 'Invalid or duplicate matrix platform'
            Assert ($entry.os -ceq $platforms[$entry.platform].runner) "Wrong runner for $($entry.platform)"
            $seen[$entry.platform] = $true
        }
        $steps = @($job.steps)
        Assert ($steps.Count -eq 9) 'Expected nine package steps'
        foreach ($step in $steps) {
            if ($step.ContainsKey('uses')) { Assert ($step.uses -cmatch '^[\w-]+/[\w-]+@[0-9a-f]{40}$') "Unpinned action: $($step.uses)" }
            if ($step.ContainsKey('run')) { $null = ReadRunScript $step.run }
        }
        AssertKeys $steps[0] @('uses') 'checkout'
        Assert ($steps[0].uses -ceq 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1') 'Expected checkout v7.0.1 pin'
        AssertKeys $steps[1] @('uses','with') 'cache'
        Assert ($steps[1].uses -like 'actions/cache@*') 'Missing cache action'
        Assert ($steps[1].with.path.Trim() -ceq "~/.cargo/registry`n~/.cargo/git`ntarget") 'Missing cargo/target cache'
        AssertKeys $steps[2] @('name','if','run') 'Linux prerequisites'
        Assert ($steps[2].name -eq 'Install Linux package tools' -and $steps[2].if -ceq "runner.os == 'Linux'") 'Invalid Linux tools step'
        AssertKeys $steps[3] @('name','if','run') 'Windows inventory'
        Assert ($steps[3].name -eq 'Record Windows build tools and licenses' -and $steps[3].if -ceq "matrix.platform == 'windows'") 'Disabled Windows inventory'
        $inventory = ReadRunScript $steps[3].run
        RequireStatement $inventory.EndBlock.Statements '$dir = ''target-evidence/T-094/package''' 'inventory directory'
        # Require an unconditional pipeline writing the inventory, not a comment,
        # string, function definition or command nested in a disabled branch.
        $writers = @($inventory.EndBlock.Statements | Where-Object {
            $_ -is [System.Management.Automation.Language.PipelineAst] -and
            (ScriptSignature $_.PipelineElements[-1].Extent.Text) -ceq (ScriptSignature 'Set-Content (Join-Path $dir ''build-inventory.txt'')')
        })
        Assert ($writers.Count -eq 1) 'Missing executable Windows inventory writer'
        $commands = @(
            './scripts/package-windows.ps1 -Verify -OutputDir target-evidence/T-094/package',
            'bash scripts/package-macos.sh --output target-evidence/T-094/package',
            'bash scripts/package-linux.sh --output target-evidence/T-094/package'
        )
        $names = @('Windows','macOS','Linux'); $ids = @('windows','macos','linux')
        for ($i = 0; $i -lt 3; $i++) {
            $step = $steps[$i + 4]
            AssertKeys $step @('name','id','continue-on-error','if','run') "package $($ids[$i])"
            Assert ($step.name -ceq $(if ($i -eq 0) { "Build and verify Windows package" } else { "Build $($names[$i]) package" }) -and $step.id -ceq $ids[$i] -and $step.'continue-on-error' -eq 'true' -and $step.if -ceq "matrix.platform == '$($ids[$i])'") "Disabled package step: $($ids[$i])"
            $ast = ReadRunScript $step.run
            $tries = @($ast.EndBlock.Statements | Where-Object { $_ -is [System.Management.Automation.Language.TryStatementAst] })
            Assert ($tries.Count -eq 1 -and $tries[0].CatchClauses.Count -eq 0) 'Expected uncaught package verification block'
            RequireStatement $tries[0].Body.Statements $commands[$i] "$($ids[$i]) package invocation"
            RequireStatement $tries[0].Body.Statements 'if ($LASTEXITCODE) { throw "Package command failed: $LASTEXITCODE" }' 'package exit guard'
        }
        $report = $steps[7]
        AssertKeys $report @('name','if','env','run') 'report step'
        Assert ($report.name -eq 'Write build identity and status report' -and $report.if -eq 'always()') 'Disabled report step'
        $bindings = @{ PACKAGE_PLATFORM = '${{ matrix.platform }}'; PACKAGE_REVISION = '${{ github.sha }}'; WINDOWS_OUTCOME = '${{ steps.windows.outcome }}'; MACOS_OUTCOME = '${{ steps.macos.outcome }}'; LINUX_OUTCOME = '${{ steps.linux.outcome }}' }
        AssertKeys $report.env @($bindings.Keys) 'report environment'
        foreach ($key in $bindings.Keys) { Assert ($report.env[$key] -ceq $bindings[$key]) "Invalid report binding: $key" }
        $ast = ReadRunScript $report.run
        foreach ($statement in @(
            '$dir = ''target-evidence/T-094/package''',
            '$outcome = @{ windows = $env:WINDOWS_OUTCOME; macos = $env:MACOS_OUTCOME; linux = $env:LINUX_OUTCOME }[$env:PACKAGE_PLATFORM]',
            '$ok = $outcome -eq ''success''',
            '$launch = $unrun',
            '$install = if ($env:PACKAGE_PLATFORM -eq ''windows'') { $operation } else { $unrun }',
            '$inventory = @(Get-ChildItem $dir -File | Where-Object Name -ne ''report.json'' | ForEach-Object { @{ name = $_.Name; size_bytes = $_.Length } })',
            'if ($artifact) { $artifactInfo = @{ name = $artifact.Name; size_bytes = $artifact.Length; sha256 = (Get-FileHash $artifact.FullName -Algorithm SHA256).Hash.ToLowerInvariant() } }',
            '$report = @{ schema = 1; platform = $env:PACKAGE_PLATFORM; os = @{windows=''windows'';macos=''macos'';linux=''linux''}[$env:PACKAGE_PLATFORM]; architecture = @{windows=''x86_64'';macos=''aarch64'';linux=''x86_64''}[$env:PACKAGE_PLATFORM]; revision = $env:PACKAGE_REVISION; artifact = $artifactInfo; inventory = $inventory; verification = @{ status = $outcome; exit_code = $(if ($ok) { 0 } else { $null }) }; operations = @{ install = $install; uninstall = $install; launch = $launch }; gpu = ''not_run''; signing = $(if ($ok -and $env:PACKAGE_PLATFORM -eq ''macos'') { ''ad_hoc'' } else { ''unsigned'' }) }',
            '$report | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $dir ''report.json'')',
            'if (-not $ok -or -not $artifact -or $artifact.Length -ge 50000000) { throw ''Package verification or 50,000,000 byte gate failed'' }'
        )) { RequireStatement $ast.EndBlock.Statements $statement 'report inventory/hash/write/failure guard' }
        $upload = $steps[8]
        AssertKeys $upload @('if','uses','with') 'artifact upload'
        Assert ($upload.if -eq 'always()' -and $upload.uses -like 'actions/upload-artifact@*') 'Disabled artifact upload'
        AssertKeys $upload.with @('name','path','if-no-files-found','retention-days') 'artifact upload options'
        Assert ($upload.with.name -ceq 'texelith-${{ matrix.platform }}-${{ github.sha }}' -and $upload.with.path -ceq 'target-evidence/T-094/package/' -and $upload.with.'if-no-files-found' -eq 'error' -and $upload.with.'retention-days' -eq '30') 'Invalid artifact upload path/identity/policy'
        return ,@()
    } catch { return ,@($_.Exception.Message) }
}
function ValidateEvidence([string]$root) {
    $reports = @(Get-ChildItem -LiteralPath $root -Filter report.json -Recurse -File)
    Assert ($reports.Count -eq 3) "Expected exactly three platform reports, found $($reports.Count)"
    $seen = @{}; $revision = $null
    foreach ($file in $reports) {
        $r = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
        Assert ($platforms.ContainsKey([string]$r.platform)) "Unknown platform in $($file.FullName)"
        $p = $platforms[[string]$r.platform]
        Assert (-not $seen.ContainsKey([string]$r.platform)) "Duplicate report for $($r.platform)"; $seen[[string]$r.platform] = $true
        Assert ($r.schema -eq 1 -and $r.os -eq $p.os -and $r.architecture -eq $p.architecture) "Invalid schema or native platform identity: $($r.platform)"
        Assert ($r.revision -match '^[0-9a-f]{40}$') "Invalid revision: $($r.platform)"
        $name = [string]$r.artifact.name
        $validName = if ($r.platform -eq 'linux') { $name -cmatch '^texelith_[0-9][0-9A-Za-z.+~:-]*_amd64\.deb$' } else { $name -ceq $p.artifact }
        Assert $validName "Invalid artifact name: $name"
        if ($null -eq $revision) { $revision = $r.revision }
        Assert ($r.revision -ceq $revision) "Reports have different revisions"
        $artifact = Join-Path $file.DirectoryName $name
        Assert (Test-Path -LiteralPath $artifact -PathType Leaf) "Missing artifact: $artifact"
        $item = Get-Item -LiteralPath $artifact; $hash = (Get-FileHash -LiteralPath $artifact -Algorithm SHA256).Hash.ToLowerInvariant()
        Assert ([long]$r.artifact.size_bytes -eq $item.Length -and $r.artifact.sha256 -eq $hash) "Artifact size/hash mismatch: $($r.platform)"
        Assert ($item.Length -lt 50000000) "Artifact exceeds 50 MB: $($r.platform)"
        Assert ($null -ne $r.inventory -and @($r.inventory).Count -gt 0) "Missing inventory: $($r.platform)"
        $inventoryNames = @{}
        foreach ($entry in $r.inventory) {
            $entryName = [string]$entry.name
            Assert ($entryName -and [IO.Path]::GetFileName($entryName) -ceq $entryName -and -not $inventoryNames.ContainsKey($entryName)) "Invalid inventory entry: $entryName"
            $inventoryNames[$entryName] = $true
            $entryPath = Join-Path $file.DirectoryName $entryName
            Assert (Test-Path -LiteralPath $entryPath -PathType Leaf) "Missing inventory file: $entryName"
            Assert ($null -ne $entry.size_bytes -and [long]$entry.size_bytes -eq (Get-Item -LiteralPath $entryPath).Length) "Inventory size mismatch: $entryName"
        }
        Assert ($inventoryNames.ContainsKey($name) -and $inventoryNames.ContainsKey('build-inventory.txt')) "Missing artifact or build inventory: $($r.platform)"
        Assert ($r.verification.status -eq 'success' -and $null -ne $r.verification.exit_code -and $r.verification.exit_code -eq 0) "Verification did not succeed: $($r.platform)"
        foreach ($key in @('install','uninstall','launch')) {
            $op = $r.operations.$key
            Assert ($null -ne $op -and $op.status -in @('pass','not_run')) "Missing $key operation: $($r.platform)"
            if ($op.status -eq 'pass') { Assert ($null -ne $op.exit_code -and $op.exit_code -eq 0) "Missing successful $key exit code" }
            else { Assert ($null -eq $op.exit_code) "Unexpected $key exit code" }
            if ($key -ne 'launch' -or $r.platform -ne 'windows') { Assert ($op.status -eq 'pass') "Required operation $key did not pass" }
        }
        Assert ($r.gpu -in @('pass','fail','not_run')) "Missing GPU status: $($r.platform)"
        Assert ($r.signing -in @('unsigned','ad_hoc','signed')) "Missing signing status: $($r.platform)"
    }
    foreach ($platform in $platforms.Keys) { Assert $seen.ContainsKey($platform) "Missing report for $platform" }
}
if ($SelfTest) {
    $yaml = Get-Content -LiteralPath $workflowPath -Raw
    $workflowErrors = ValidateWorkflow $yaml
    Assert ($workflowErrors.Count -eq 0) ("Workflow validation failed: " + ($workflowErrors -join '; '))
    foreach ($mutation in @($yaml.Replace('macos-latest','windows-2022'), $yaml.Replace('actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1','actions/checkout@v4'), $yaml.Replace('target-evidence/T-094/package','tmp/package'), $yaml.Replace('runs-on: ${{ matrix.os }}','runs-on: windows-2022'), $yaml.Replace("if: matrix.platform == 'linux'", "if: false"), $yaml.Replace('bash scripts/package-linux.sh --output','bash missing.sh --output'), $yaml.Replace("tags: ['v*']", "branches: ['main']`n    tags: ['v*']"), $yaml.Replace('uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9', "id: cache`n        uses: actions/cache@v4"), $yaml.Replace('  package:', "  package:`n    if: false"))) { Assert ((ValidateWorkflow $mutation).Count -gt 0) 'Negative workflow fixture was accepted' }
    $fixtures = Join-Path $repo 'docs/evidence/T-094/negative-fixtures'
    $workflowFixtures = Get-Content -LiteralPath (Join-Path $fixtures 'workflow-mutations.json') -Raw | ConvertFrom-Json
    foreach ($fixture in $workflowFixtures) {
        # Preserve the historical negative cases against the current build-only
        # command/report spellings; no mutation or expected rejection is removed.
        $fixture.target = $fixture.target.Replace('bash scripts/package-macos.sh --verify --output', 'bash scripts/package-macos.sh --output').Replace('xvfb-run -a bash scripts/package-linux.sh --verify --output', 'bash scripts/package-linux.sh --output').Replace('operations = @{ install = $operation; uninstall = $operation; launch = $launch }', 'operations = @{ install = $install; uninstall = $install; launch = $launch }').Replace('Build and verify macOS package','Build macOS package').Replace('Build and verify Linux package','Build Linux package')
        $pattern = if ($fixture.kind -like '*-step') {
            '(?ms)^      - name: ' + [regex]::Escape($fixture.target) + '\r?\n.*?(?=^      - |\z)'
        } else { '(?m)^ *' + [regex]::Escape($fixture.target) + '\r?$' }
        $matchesFound = [regex]::Matches($yaml, $pattern)
        Assert ($matchesFound.Count -eq 1) "Stale workflow fixture: $($fixture.name)"
        $replacement = if ($fixture.kind -like 'comment-*') { [regex]::Replace($matchesFound[0].Value, '(?m)^( *)(\S.*)$', '$1# $2') } else { '' }
        $mutation = $yaml.Substring(0, $matchesFound[0].Index) + $replacement + $yaml.Substring($matchesFound[0].Index + $matchesFound[0].Length)
        $rejected = ValidateWorkflow $mutation
        Assert ($rejected.Count -gt 0 -and ($rejected -join '; ').Contains($fixture.error)) "Negative workflow fixture accepted or rejected for wrong reason: $($fixture.name): $rejected"
    }
    # Comments and harmless formatting must remain valid; strings and disabled
    # branches must not stand in for executable statements in the correct step.
    $positive = $yaml.Replace('name: Native package artifacts', 'name: Native package artifacts # harmless comment').Replace('    runs-on:', '    # runner comment' + "`n" + '    runs-on:').Replace('actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0', 'actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # uses: ignored/unpinned@v1')
    $positive = $positive.Replace('            bash scripts/package-linux.sh', "            # Packaging comment`n            bash scripts/package-linux.sh")
    Assert ((ValidateWorkflow $positive).Count -eq 0) 'Valid YAML comments were rejected'
    $command = 'bash scripts/package-linux.sh --output target-evidence/T-094/package'
    $guard = 'if (-not $ok -or -not $artifact -or $artifact.Length -ge 50000000) { throw ''Package verification or 50,000,000 byte gate failed'' }'
    foreach ($mutation in @(
        $yaml.Replace($command, "'$command'"),
        $yaml.Replace($command, "if (`$false) { $command }"),
        $yaml.Replace($guard, "if (`$false) { $guard }"),
        $yaml.Replace("- name: Record Windows build tools and licenses`n        if: matrix.platform == 'windows'", "- name: Record Windows build tools and licenses`n        if: false").Replace("- name: Record Windows build tools and licenses`r`n        if: matrix.platform == 'windows'", "- name: Record Windows build tools and licenses`r`n        if: false"),
        $yaml.Replace('    runs-on:', "    runs-on: windows-2022`n    runs-on:"),
        $yaml.Replace('        run: |', '        run: >')
    )) { Assert ($mutation -cne $yaml -and (ValidateWorkflow $mutation).Count -gt 0) 'Disabled/invalid structured workflow fixture was accepted' }
    foreach ($name in @('missing-platform.json','bad-hash.json')) { Assert (Test-Path -LiteralPath (Join-Path $fixtures $name)) "Missing negative evidence fixture $name" }
    $sample = Join-Path ([IO.Path]::GetTempPath()) ('T094-' + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $sample | Out-Null
    try {
        foreach ($platform in $platforms.Keys) {
            $dir = Join-Path $sample $platform; New-Item -ItemType Directory -Path $dir | Out-Null
            $p = $platforms[$platform]; $artifact = Join-Path $dir $p.artifact; Set-Content -LiteralPath $artifact -Value 'fixture' -NoNewline
            $item = Get-Item -LiteralPath $artifact
            Set-Content (Join-Path $dir 'build-inventory.txt') 'tools'
            $report = @{ schema = 1; platform = $platform; os = $p.os; architecture = $p.architecture; revision = ('a' * 40); artifact = @{ name = $p.artifact; size_bytes = $item.Length; sha256 = (Get-FileHash $artifact -Algorithm SHA256).Hash.ToLowerInvariant() }; inventory = @(@{ name = $p.artifact; size_bytes = $item.Length }, @{ name = 'build-inventory.txt'; size_bytes = (Get-Item (Join-Path $dir 'build-inventory.txt')).Length }); verification = @{ status = 'success'; exit_code = 0 }; operations = @{ install = @{ status = 'pass'; exit_code = 0 }; uninstall = @{ status = 'pass'; exit_code = 0 }; launch = @{ status = 'pass'; exit_code = 0 } }; gpu = 'not_run'; signing = 'unsigned' }
            $report | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 (Join-Path $dir 'report.json')
        }
        ValidateEvidence $sample
        $linuxReport = Join-Path (Join-Path $sample 'linux') 'report.json'; $linuxJson = Get-Content $linuxReport -Raw; Remove-Item -LiteralPath $linuxReport
        $failed = $false; try { ValidateEvidence $sample } catch { $failed = $true }; Assert $failed 'Missing-platform report fixture was accepted'
        Set-Content -LiteralPath $linuxReport -Value $linuxJson
        $windowsReport = Join-Path (Join-Path $sample 'windows') 'report.json'
        $original = Get-Content $windowsReport -Raw
        foreach ($kind in @('inventory','missing-file','exit-code')) {
            $r = $original | ConvertFrom-Json
            if ($kind -eq 'inventory') { $r.inventory = @() }
            if ($kind -eq 'missing-file') { $r.inventory[1].name = 'missing.txt' }
            if ($kind -eq 'exit-code') { $r.operations.install.exit_code = $null }
            $r | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $windowsReport
            $failed = $false; try { ValidateEvidence $sample } catch { $failed = $true }; Assert $failed "Negative $kind fixture was accepted"
        }
        Set-Content -LiteralPath $windowsReport -Value $original
        $windowsReport = Join-Path (Join-Path $sample 'windows') 'report.json'; $r = Get-Content $windowsReport -Raw | ConvertFrom-Json; $r.artifact.sha256 = '0' * 64; $r | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $windowsReport
        $failed = $false; try { ValidateEvidence $sample } catch { $failed = $true }; Assert $failed 'Bad-hash report fixture was accepted'
    } finally {
        $resolvedSample = [IO.Path]::GetFullPath($sample)
        $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        Assert ($resolvedSample.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and [IO.Path]::GetFileName($resolvedSample) -like 'T094-*') 'Unsafe fixture cleanup path'
        Remove-Item -LiteralPath $resolvedSample -Recurse -Force
    }
    Write-Output "PASS: structured matrix/steps, pinned actions, executable commands/guards, $($workflowFixtures.Count) workflow regression fixtures and evidence fixtures"
    exit 0
}
if ($EvidenceRoot) { ValidateEvidence $(if ([IO.Path]::IsPathRooted($EvidenceRoot)) { [IO.Path]::GetFullPath($EvidenceRoot) } else { [IO.Path]::GetFullPath((Join-Path $repo $EvidenceRoot)) }); Write-Output 'PASS: all native package evidence validated'; exit 0 }
$errors = ValidateWorkflow (Get-Content -LiteralPath $workflowPath -Raw)
Assert ($errors.Count -eq 0) ($errors -join '; ')
Write-Output 'PASS: native Windows, macOS and Linux packaging jobs configured; build/hash reports are generated per platform; hosted launch/GPU checks are not run.'
