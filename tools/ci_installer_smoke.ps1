# CI smoke test of the finished installer on a Windows runner (see .github/workflows/installer.yml):
#   silent install -> what was installed (and where) -> the installed console exe (check, about, FFmpeg, expiry/heads
#   refusal) -> the window starts, draws, answers and closes (screenshots) -> silent uninstall leaves nothing.
param(
  [Parameter(Mandatory)] [string]$Installer,
  [Parameter(Mandatory)] [string]$OutDir,
  [Parameter(Mandatory)] [string]$Product,
  [Parameter(Mandatory)] [string]$CliExe,
  [Parameter(Mandatory)] [string]$GuiExe,
  [Parameter(Mandatory)] [string]$AppGuid,
  [Parameter(Mandatory)] [string]$Version,
  [string]$Expires = "",  # empty: a paid build that never expires
  [switch]$RealHeads
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$script:problems = @()
function Check([bool]$ok, [string]$what) {
  if ($ok) { Write-Host "  ok    $what" } else { Write-Host "  FAIL  $what"; $script:problems += $what }
}
function Step([string]$name) { Write-Host ""; Write-Host "== $name" }

$installDir = Join-Path $env:LOCALAPPDATA "Programs\$Product"
$dataDir = Join-Path $env:LOCALAPPDATA $Product
$shortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$Product.lnk"
$uninstKey = "Software\Microsoft\Windows\CurrentVersion\Uninstall\{$AppGuid}_is1"
$cli = Join-Path $installDir "$CliExe.exe"
$gui = Join-Path $installDir "$GuiExe.exe"

Step "silent install (no admin switch, per user)"
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process $Installer -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", "/LOG=`"$OutDir\install.log`"" -Wait -PassThru
Check ($p.ExitCode -eq 0) "installer exit code 0 (got $($p.ExitCode)) in $([int]$sw.Elapsed.TotalSeconds) s"
Check (Test-Path $cli) "$CliExe.exe installed under %LOCALAPPDATA%\Programs"
Check (Test-Path $gui) "$GuiExe.exe installed"
Check (Test-Path (Join-Path $installDir "ffmpeg\bin\ffmpeg.exe")) "bundled FFmpeg installed"
Check (Test-Path (Join-Path $installDir "heads\fast_h6.txt")) "risk heads installed"
Check (Test-Path (Join-Path $installDir "obs\streamsafe_guard.lua")) "OBS guard script installed"
foreach ($f in "NOTICE.txt", "NVIDIA-Open-Model-License.txt", "Apache-2.0.txt", "ONNX-Runtime-MIT.txt", "NOTICE.md") {
  Check (Test-Path (Join-Path $installDir "licenses\$f")) "licence file $f"
}
Check (Test-Path (Join-Path $installDir "ffmpeg\LICENSE.txt")) "FFmpeg LICENSE.txt"
Check (Test-Path $shortcut) "Start-menu shortcut"
Check (Test-Path (Join-Path $installDir "unins000.exe")) "uninstaller"
Check (Test-Path "HKCU:\$uninstKey") "uninstall entry is per user (HKCU)"
Check (-not (Test-Path "HKLM:\$uninstKey")) "nothing in HKLM (64-bit view)"
Check (-not (Test-Path "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{$AppGuid}_is1")) "nothing in HKLM (32-bit view)"
Check (-not (Test-Path (Join-Path $env:ProgramFiles $Product))) "nothing in Program Files"
Check (-not (Test-Path (Join-Path $env:ProgramData $Product))) "nothing in ProgramData"
$run = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -ErrorAction SilentlyContinue
Check (-not ($run -and ($run.PSObject.Properties.Name -match $Product))) "no autostart (Run) entry"
Check (-not (Get-Service | Where-Object { $_.Name -match $Product -or $_.DisplayName -match $Product })) "no service"
Check (-not (Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object { $_.TaskName -match $Product })) "no scheduled task"
$size = (Get-ChildItem $installDir -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("  installed size {0:N0} MB, installer {1:N0} MB" -f ($size / 1MB), ((Get-Item $Installer).Length / 1MB))
Get-ChildItem $installDir | Select-Object Name, Length | Format-Table | Out-String | Write-Host

Step "version information of the exe"
foreach ($e in $cli, $gui) {
  $vi = (Get-Item $e).VersionInfo
  Check ($vi.ProductName -eq $Product -and $vi.FileVersion -eq $Version -and $vi.CompanyName) "$(Split-Path $e -Leaf): ProductName=$($vi.ProductName) FileVersion=$($vi.FileVersion) Company=$($vi.CompanyName)"
}

Step "installed console program"
$aboutText = (& $cli about --json | Out-String)
Write-Host $aboutText
$about = $aboutText | ConvertFrom-Json
Check ($about.frozen -eq $true) "runs as the packaged program"
if ($Expires) {
  Check ($about.expires -eq $Expires -and -not $about.expired -and $about.days_left -gt 0) "beta ends $($about.expires), $($about.days_left) days left"
} else {
  Check (-not $about.expires -and -not $about.expired) "no end date (paid build)"
}
Check ($about.ffmpeg -like "$installDir*") "FFmpeg in use is the bundled one: $($about.ffmpeg)"
$tr = (& $cli translate-selftest | Out-String)
Write-Host $tr
Check ($LASTEXITCODE -eq 0 -and $tr -match "TRANSLATE_LIBS_OK") "translation libraries load in the packaged program"
Check ($about.data_dir -like "$env:LOCALAPPDATA*") "data folder is per user: $($about.data_dir)"
Check ($about.version -eq $Version) "version $($about.version)"

$checkText = (& $cli check --json | Out-String)
$checkCode = $LASTEXITCODE
$checkText | Out-File "$OutDir\check.json" -Encoding utf8
$check = $checkText | ConvertFrom-Json
foreach ($c in $check.checks) { Write-Host ("  [{0,-4}] {1,-8} {2}" -f $c.status, $c.name, $c.detail) }
$ff = $check.checks | Where-Object { $_.name -eq "ffmpeg" }
Check ($ff.status -eq "ok") "check --json: FFmpeg OK (exit code $checkCode, verdict $($check.verdict))"
$rt = $check.checks | Where-Object { $_.name -eq "runtime" }
Check ($rt.status -ne "fail") "check --json: GPU runtime loads ($($rt.detail))"

$ffmpeg = Join-Path $installDir "ffmpeg\bin\ffmpeg.exe"
$ver = (& $ffmpeg -version | Out-String)
$ver | Out-File "$OutDir\ffmpeg-version.txt" -Encoding utf8
Check ($ver -notmatch "--enable-(gpl|nonfree|libx264|libx265)") "bundled FFmpeg is LGPL (no --enable-gpl/nonfree/libx264)"
$prot = (& $ffmpeg -hide_banner -protocols | Out-String)
Check ($prot -match "(?m)^\s*srt\s*$") "FFmpeg has the srt protocol (OBS to the app)"
Check ($prot -match "(?m)^\s*rtmps?\s*$") "FFmpeg has rtmp"
Check ($prot -match "(?m)^\s*rtmps\s*$") "FFmpeg has rtmps (TLS) for the platform"
$encText = (& $cli encoder-probe | Out-String)
Write-Host $encText
if ($encText -notmatch '"chosen": "[a-z0-9_]+"') { Write-Host "::warning::no H.264 encoder works on this runner (a PC with a graphics card or Media Foundation has one)" }

Step "the live path of the installed program: OBS over SRT -> engine -> RTMP (selftest-stream: no models, every frame covered)"
$ffmpeg = Join-Path $installDir "ffmpeg\bin\ffmpeg.exe"
if ($encText -match '"chosen": "[a-z0-9_]+"') {
  $live = Join-Path $OutDir "live"; New-Item -ItemType Directory -Force $live | Out-Null
  $flv = Join-Path $live "platform.flv"
  # the "platform": FFmpeg listening for one RTMP publisher
  $plat = Start-Process $ffmpeg -ArgumentList "-hide_banner","-loglevel","error","-y","-listen","1","-i","rtmp://127.0.0.1:19350/app/test_key_123","-c","copy",$flv -PassThru -WindowStyle Hidden
  Start-Sleep -Seconds 2
  $eng = Start-Process $cli -ArgumentList "selftest-stream","--input","srt://127.0.0.1:19000?mode=listener","--output","rtmp://127.0.0.1:19350/app/test_key_123","--delay","1","--size","640x360","--report-dir",$live `
    -RedirectStandardOutput (Join-Path $live "engine_out.txt") -RedirectStandardError (Join-Path $live "engine_err.txt") -PassThru -WindowStyle Hidden
  $end = (Get-Date).AddSeconds(60); $up = $false
  while ((Get-Date) -lt $end -and -not $eng.HasExited) {
    if ((Get-Content (Join-Path $live "engine_out.txt") -Raw -ErrorAction SilentlyContinue) -match '"waiting_obs"') { $up = $true; break }
    Start-Sleep -Milliseconds 500
  }
  Check $up "the SRT listener is up (STATUS waiting_obs)"
  # OBS stand-in, twice (OBS stops and starts again): the engine must take both without a restart
  foreach ($round in 1, 2) {
    & $ffmpeg -hide_banner -loglevel error -re -f lavfi -i "testsrc2=size=640x360:rate=30:duration=6" -f lavfi -i "sine=frequency=440:duration=6" `
      -c:v $(if ($encText -match '"chosen": "([a-z0-9_]+)"') { $Matches[1] } else { "h264_mf" }) -g 30 -c:a aac -f mpegts "srt://127.0.0.1:19000?mode=caller"
    Start-Sleep -Seconds 2
  }
  Start-Sleep -Seconds 3
  # the whole tree: the engine's own FFmpeg and analyzer processes too (else they hold files of the program folder)
  if (-not $eng.HasExited) { & taskkill /T /F /PID $eng.Id | Out-Null }
  if (-not $plat.HasExited) { Start-Sleep -Seconds 3; if (-not $plat.HasExited) { & taskkill /T /F /PID $plat.Id | Out-Null } }
  Start-Sleep -Seconds 2
  Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($installDir) } | ForEach-Object { Write-Host "left over: $($_.Name) $($_.Id)"; Stop-Process -Id $_.Id -Force }
  $out = Get-Content (Join-Path $live "engine_out.txt") -Raw -ErrorAction SilentlyContinue
  Write-Host ($out -split "`n" | Where-Object { $_ -match "^STATUS" } | Out-String)
  Check (([regex]::Matches($out, '"obs_connected"')).Count -ge 2) "OBS connected twice (stop and start again without a restart)"
  Check ($out -match '"platform_connected"') "the platform (local RTMP) took the stream"
  Check ((Test-Path $flv) -and (Get-Item $flv).Length -gt 20000) "the platform received video ($(if (Test-Path $flv) { (Get-Item $flv).Length } else { 0 }) bytes)"
  Check ($out -notmatch "test_key_123") "the stream key is never printed"
} else {
  Write-Host "::warning::no H.264 encoder on this runner: the live path smoke is skipped"
}

Step "real OBS: the program sets OBS up by itself, opens it, and OBS streams to the program without a click"
if ($encText -match '"chosen": "[a-z0-9_]+"') {
  $ro = Join-Path $OutDir "real_obs"; New-Item -ItemType Directory -Force $ro | Out-Null
  & choco install obs-studio -y --no-progress *> (Join-Path $ro "choco.txt")
  $obsExe = "C:\Program Files\obs-studio\bin\64bit\obs64.exe"
  if (Test-Path $obsExe) {
    # a streamer's OBS after its first run: one profile (Twitch selected, a key in it), a scene, small video for the runner
    $cfg = Join-Path $env:APPDATA "obs-studio"; $prof = Join-Path $cfg "basic\profiles\Untitled"
    New-Item -ItemType Directory -Force $prof, (Join-Path $cfg "basic\scenes") | Out-Null
    $glob = "[General]`r`nFirstRun=false`r`nLastVersion=520093696`r`n`r`n[Basic]`r`nProfile=Untitled`r`nProfileDir=Untitled`r`nSceneCollection=Untitled`r`nSceneCollectionFile=Untitled`r`nConfigOnNewProfile=false`r`n"
    $glob | Out-File (Join-Path $cfg "global.ini") -Encoding utf8; $glob | Out-File (Join-Path $cfg "user.ini") -Encoding utf8
    "[General]`r`nName=Untitled`r`n`r`n[Video]`r`nBaseCX=640`r`nBaseCY=360`r`nOutputCX=640`r`nOutputCY=360`r`nFPSType=0`r`nFPSCommon=30`r`n`r`n[Output]`r`nMode=Simple`r`n`r`n[SimpleOutput]`r`nVBitrate=2500`r`nStreamEncoder=x264`r`n" | Out-File (Join-Path $prof "basic.ini") -Encoding utf8
    '{"type":"rtmp_common","settings":{"service":"Twitch","server":"auto","key":"live_123"}}' | Out-File (Join-Path $prof "service.json") -Encoding ascii
    # a saved scene (OBS only honours --startstreaming when it loads a saved scene collection, as a streamer's OBS does)
    '{"name":"Untitled","current_scene":"Scene","current_program_scene":"Scene","scene_order":[{"name":"Scene"}],"sources":[' +
      '{"id":"color_source_v3","versioned_id":"color_source_v3","name":"Game","uuid":"0e4f5b2e-1111-4c1a-9a7e-000000000001","settings":{"color":4278222848,"width":640,"height":360}},' +
      '{"id":"scene","versioned_id":"scene","name":"Scene","uuid":"0e4f5b2e-1111-4c1a-9a7e-000000000002","settings":{"id_counter":1,"items":[{"name":"Game","source_uuid":"0e4f5b2e-1111-4c1a-9a7e-000000000001","visible":true,"id":1}]}}]}' |
      Out-File (Join-Path $cfg "basic\scenes\Untitled.json") -Encoding ascii
    $flv2 = Join-Path $ro "platform.flv"
    $plat = Start-Process $ffmpeg -ArgumentList "-hide_banner","-loglevel","error","-y","-listen","1","-i","rtmp://127.0.0.1:19350/app/test_key_123","-c","copy",$flv2 -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 2
    $eng = Start-Process $cli -ArgumentList "selftest-stream","--input","srt://127.0.0.1:9000?mode=listener","--output","rtmp://127.0.0.1:19350/app/test_key_123","--delay","1","--size","640x360","--report-dir",$ro `
      -RedirectStandardOutput (Join-Path $ro "engine_out.txt") -RedirectStandardError (Join-Path $ro "engine_err.txt") -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 8
    $setup = (& $cli obs-setup --open | Out-String); Write-Host $setup
    Check ($setup -match '"result": "ok"' -and $setup -match '"set": true' -and $setup -match '"opened": true') "obs-setup: OBS set up and opened"
    $svc = Get-Content (Join-Path $prof "service.json") -Raw; Write-Host $svc
    Check ($svc -match "rtmp_custom" -and $svc -match "srt://127.0.0.1:9000") "OBS's stream settings now point at the program"
    $end = (Get-Date).AddSeconds(120); $conn = $false
    while ((Get-Date) -lt $end) {
      if ((Get-Content (Join-Path $ro "engine_out.txt") -Raw -ErrorAction SilentlyContinue) -match '"obs_connected"') { $conn = $true; break }
      Start-Sleep -Seconds 1
    }
    Start-Sleep -Seconds 10
    try { Add-Type -AssemblyName System.Drawing; $bmp = New-Object System.Drawing.Bitmap 1280, 800; $g = [System.Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen(0, 0, 0, 0, $bmp.Size); $bmp.Save((Join-Path $ro "screen.png")) } catch { Write-Host "no screenshot: $_" }
    Get-Process obs64 -ErrorAction SilentlyContinue | Stop-Process -Force
    if (-not $eng.HasExited) { & taskkill /T /F /PID $eng.Id | Out-Null }
    if (-not $plat.HasExited) { Start-Sleep -Seconds 3; if (-not $plat.HasExited) { & taskkill /T /F /PID $plat.Id | Out-Null } }
    Start-Sleep -Seconds 2
    Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($installDir) } | ForEach-Object { Stop-Process -Id $_.Id -Force }
    Copy-Item (Join-Path $env:APPDATA "obs-studio\logs") (Join-Path $ro "obs_logs") -Recurse -ErrorAction SilentlyContinue
    $o = Get-Content (Join-Path $ro "engine_out.txt") -Raw -ErrorAction SilentlyContinue
    Write-Host ($o -split "`n" | Where-Object { $_ -match "^STATUS" } | Select-Object -First 12 | Out-String)
    Check $conn "real OBS connected to the program by itself (--startstreaming)"
    Check ((Test-Path $flv2) -and (Get-Item $flv2).Length -gt 20000) "the platform received the stream from real OBS ($(if (Test-Path $flv2) { (Get-Item $flv2).Length } else { 0 }) bytes)"
  } else {
    Write-Host "::warning::OBS could not be installed on this runner: the real OBS check is skipped"
  }
}

Step "the engine refuses to stream with test heads (fail closed)"
if (-not $RealHeads) {
  $out = (& $cli run --input "device:none" --output window | Out-String)
  Check ($LASTEXITCODE -eq 5 -and $out -match "HEADS_PLACEHOLDER") "run exits 5 with HEADS_PLACEHOLDER (got $LASTEXITCODE)"
}
if ($RealHeads) {
  $out = (& $cli run --input "device:none" --output window | Out-String)
  Check ($LASTEXITCODE -eq 6 -and $out -match "WEAK_GPU") "no gaming graphics card on the runner: run exits 6 with WEAK_GPU (got $LASTEXITCODE)"
}

Step "the window"
Add-Type -AssemblyName System.Drawing, System.Windows.Forms
Add-Type @"
using System; using System.Runtime.InteropServices;
public class Win { [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c); }
"@
function Shot([string]$lang, [string]$tab, [string]$name, [int]$waitS) {
  New-Item -ItemType Directory -Force $dataDir | Out-Null
  '{"lang": "' + $lang + '"}' | Out-File (Join-Path $dataDir "settings.json") -Encoding ascii
  $env:STREAMSAFE_TAB = $tab
  $w = Start-Process $gui -PassThru
  Remove-Item Env:STREAMSAFE_TAB
  $end = (Get-Date).AddSeconds(40)
  while ((Get-Date) -lt $end -and $w.MainWindowHandle -eq 0 -and -not $w.HasExited) { Start-Sleep -Milliseconds 300; $w.Refresh() }
  Check (-not $w.HasExited -and $w.MainWindowHandle -ne 0) "$name`: window opened"
  if ($w.HasExited) { return }
  Check ($w.MainWindowTitle -eq $Product) "$name`: title is '$($w.MainWindowTitle)'"
  [Win]::ShowWindow($w.MainWindowHandle, 9) | Out-Null
  [Win]::SetForegroundWindow($w.MainWindowHandle) | Out-Null
  Start-Sleep -Seconds $waitS   # the system check runs in the background and fills the list
  $w.Refresh()
  Check ($w.Responding) "$name`: window answers"
  $b = [Windows.Forms.Screen]::PrimaryScreen.Bounds
  $bmp = New-Object Drawing.Bitmap $b.Width, $b.Height
  $g = [Drawing.Graphics]::FromImage($bmp)
  $g.CopyFromScreen($b.Location, [Drawing.Point]::Empty, $b.Size)
  $bmp.Save((Join-Path $OutDir "$name.png"), [Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose(); $bmp.Dispose()
  $null = $w.CloseMainWindow()
  $closed = $w.WaitForExit(20000)
  Check $closed "$name`: window closes when asked"
  if (-not $closed) { $w.Kill() }
  Check ((Get-Process -Name $CliExe, $GuiExe -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) "$name`: no process left behind"
}
Shot "ar" "home" "ui_ar_home" 20
Shot "ar" "check" "ui_ar_check" 20
Shot "ar" "settings" "ui_ar_settings" 6
Shot "en" "home" "ui_en_home" 6
Shot "en" "check" "ui_en_check" 6
Check (Test-Path (Join-Path $dataDir "settings.json")) "settings are in the per-user data folder"

Step "silent uninstall"
$un = Join-Path $installDir "unins000.exe"
Start-Process $un -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=`"$OutDir\uninstall.log`"" -Wait
$end = (Get-Date).AddSeconds(90)   # Inno's uninstaller hands over to a copy of itself; wait for it to finish
while ((Get-Date) -lt $end -and (Test-Path $installDir)) { Start-Sleep -Seconds 1 }
Check (-not (Test-Path $installDir)) "program folder is gone"
if (Test-Path $installDir) { Get-ChildItem $installDir -Recurse | Select-Object FullName | Format-Table | Out-String | Write-Host }
Check (-not (Test-Path $shortcut)) "Start-menu shortcut is gone"
Check (-not (Test-Path "HKCU:\$uninstKey")) "uninstall entry is gone"
Check (-not (Test-Path $dataDir)) "user data folder is gone (silent uninstall; /KEEPDATA keeps it)"

Write-Host ""
if ($script:problems.Count) { Write-Host "::error::$($script:problems.Count) check(s) failed: $($script:problems -join ' | ')"; exit 1 }
Write-Host "installer smoke: all checks passed"
exit 0  # a native command earlier may leave $LASTEXITCODE non-zero (the check exit code 2 is expected)
