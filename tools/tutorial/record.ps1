# Records the tutorial screen on a Windows runner: a streamer's OBS (first run done, empty scene), the platform
# stand-in, the screen recorder, then the real window driven by tools/tutorial/drive_app.py.
$ErrorActionPreference = "Continue"
$out = Join-Path $PWD "tutorial_out"; New-Item -ItemType Directory -Force $out | Out-Null
Add-Type -AssemblyName System.Windows.Forms
$scr = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; Write-Host "screen $($scr.Width)x$($scr.Height)"

# OBS after its first run: one profile, an empty scene (the program adds the screen capture)
$cfg = Join-Path $env:APPDATA "obs-studio"; $prof = Join-Path $cfg "basic\profiles\Untitled"
New-Item -ItemType Directory -Force $prof, (Join-Path $cfg "basic\scenes") | Out-Null
$glob = "[General]`r`nFirstRun=false`r`nLastVersion=520093696`r`n`r`n[Basic]`r`nProfile=Untitled`r`nProfileDir=Untitled`r`nSceneCollection=Untitled`r`nSceneCollectionFile=Untitled`r`nConfigOnNewProfile=false`r`n"
$glob | Out-File (Join-Path $cfg "global.ini") -Encoding utf8; $glob | Out-File (Join-Path $cfg "user.ini") -Encoding utf8
"[General]`r`nName=Untitled`r`n`r`n[Video]`r`nBaseCX=1280`r`nBaseCY=720`r`nOutputCX=1280`r`nOutputCY=720`r`nFPSType=0`r`nFPSCommon=30`r`n`r`n[Output]`r`nMode=Simple`r`n`r`n[SimpleOutput]`r`nVBitrate=6000`r`nStreamEncoder=x264`r`n" | Out-File (Join-Path $prof "basic.ini") -Encoding utf8
'{"type":"rtmp_common","settings":{"service":"Twitch","server":"auto","key":""}}' | Out-File (Join-Path $prof "service.json") -Encoding ascii
'{"name":"Untitled","current_scene":"Scene","current_program_scene":"Scene","scene_order":[{"name":"Scene"}],"sources":[{"id":"scene","versioned_id":"scene","name":"Scene","uuid":"0e4f5b2e-1111-4c1a-9a7e-000000000009","settings":{"id_counter":0,"items":[]}}]}' |
  Out-File (Join-Path $cfg "basic\scenes\Untitled.json") -Encoding ascii

$ff = (Get-Command ffmpeg).Source
$plat = Start-Process $ff -ArgumentList "-hide_banner","-loglevel","error","-y","-listen","1","-i","rtmp://127.0.0.1:19350/app/demo","-c","copy",(Join-Path $out "platform.flv") -PassThru -WindowStyle Hidden
# the screen recorder: stdin open so it can be stopped cleanly with "q"
$psi = New-Object System.Diagnostics.ProcessStartInfo $ff
$psi.Arguments = "-hide_banner -loglevel error -y -f gdigrab -framerate 30 -draw_mouse 1 -i desktop -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p `"$out\screen.mp4`""
$psi.RedirectStandardInput = $true; $psi.UseShellExecute = $false
$rec = [System.Diagnostics.Process]::Start($psi)
Start-Sleep -Seconds 2
python tools/tutorial/drive_app.py $out 2>&1 | Tee-Object (Join-Path $out "drive.txt")
Start-Sleep -Seconds 1
$rec.StandardInput.Write("q"); $rec.WaitForExit(20000) | Out-Null
Get-Process obs64 -ErrorAction SilentlyContinue | Stop-Process -Force
if (-not $plat.HasExited) { & taskkill /T /F /PID $plat.Id | Out-Null }
Copy-Item (Join-Path $cfg "logs") (Join-Path $out "obs_logs") -Recurse -ErrorAction SilentlyContinue
Get-ChildItem $out | Format-Table Name, Length
