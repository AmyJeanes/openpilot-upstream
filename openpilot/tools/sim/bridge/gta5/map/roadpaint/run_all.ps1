# Whole-map pipeline at low priority: roadpaint (tiles, segments, decals) -> rpchain (polylines) -> features ->
# per-link survey -> comparison with the camera survey -> prioritised disagreements. Logs to run_all.log.
param([string]$Out = "rp_all", [string]$Game = "D:\SteamLibrary\steamapps\common\Grand Theft Auto V Enhanced")
$G = $PSScriptRoot
$log = "$G\run_all.log"
function Step($name, $exe, $argList) {
  "[$(Get-Date -Format HH:mm:ss)] $name" | Add-Content $log
  $p = Start-Process -FilePath $exe -ArgumentList $argList -RedirectStandardError "$G\$Out.$name.err" -RedirectStandardOutput "$G\$Out.$name.out" -NoNewWindow -PassThru
  $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()
  "[$(Get-Date -Format HH:mm:ss)] $name exit $($p.ExitCode)" | Add-Content $log
  if ($p.ExitCode -ne 0) { throw "$name failed" }
}
$env:RP_THREADS = "6"; $env:RP_DUMPTEX = "0"; $env:RP_ATLAS = "$G\atlas_cells.json"
Remove-Item $log -ErrorAction SilentlyContinue; Set-Content $log ""
Step "roadpaint" "$G\roadpaint\bin\Release\net8.0-windows\roadpaint.exe" @("`"$Game`"", "$G\$Out")
Step "rpchain" "$G\rpchain\bin\Release\net8.0\rpchain.exe" @("$G\$Out")
$W = "/mnt/" + $G.Substring(0, 1).ToLower() + ($G.Substring(2) -replace '\\', '/')  # the same folder from WSL
$py = "GTA5MAP=~/gta5map_lanes nice -n 15 ~/git/openpilot-slowroads/.venv/bin/python3 -W ignore"
Step "features" "wsl.exe" @("-e", "bash", "-lc", "`"cd $W && $py features.py $Out atlas_cells.json`"")
Step "survey" "wsl.exe" @("-e", "bash", "-lc", "`"cd $W && $py survey_rp.py $Out $Out/survey_gf.jsonl --workers 6`"")
Step "compare" "wsl.exe" @("-e", "bash", "-lc", "`"cd $W && rm -rf disagreement_img && $py compare_trips.py $Out/survey_gf.jsonl ~/gta5test/map_audit/survey/trips.jsonl disagreements.jsonl --img disagreement_img --tiles $Out/tiles`"")
Step "prioritise" "wsl.exe" @("-e", "bash", "-lc", "`"cd $W && $py prioritise.py disagreements.jsonl $Out/features.jsonl disagreements_top.jsonl`"")
"[$(Get-Date -Format HH:mm:ss)] all done" | Add-Content $log
