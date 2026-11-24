# Builds gta5op.asi (the loader) and the core with Visual Studio's CMake and Ninja, and optionally installs them into the game.
#
# -Install puts each core in <game>\gta5op\core under a name of its own, gta5op_core_<build time>[_<Label>].dll: a new
# file every time, never one written over, because the game has been seen to hold every file that was in its folder when
# it started against writing and deleting. The loader runs the newest by name and swaps to a newer one within a second
# of it appearing, game running or not. The newest three are kept. To go back to an older core, install it again with
# -Core <file>: it gets a new name, so it is the newest.
# gta5op.asi itself only changes with the game closed.
# The plugin's log is the newest file in <game>\gta5op\logs (gta5op_<game start>.log, one for each run of the game).
param(
  [string]$GameDir = "D:\SteamLibrary\steamapps\common\Grand Theft Auto V Enhanced",
  [switch]$Install,
  [string]$Label,   # a word for this build: in its file name, the log's init line and the state's plugin.build
  [string]$Backup,  # with -Install, keeps the core this one replaces (and a replaced loader) as gta5op\backups\<file>.bak-pre-<Backup>-<date>
  [string]$Core     # with -Install, installs this core file rather than building one
)
$ErrorActionPreference = "Stop"
if ($Label -and $Label -notmatch '^[A-Za-z0-9.-]+$') { throw "-Label: letters, digits, dots and hyphens only" }
$version = (Get-Date -Format "yyyyMMdd-HHmmss") + $(if ($Label) { "_$Label" })
$build = Join-Path $env:LOCALAPPDATA "gta5op\build"

$built = -not $Core
if ($built) {
  $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -prerelease -products * -property installationPath
  Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
  Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null

  # MSVC and CMake handle \wsl.localhost sources poorly; build from a local copy
  $src = Join-Path $env:LOCALAPPDATA "gta5op\src"
  New-Item -ItemType Directory -Force $src, $build | Out-Null
  robocopy $PSScriptRoot $src /MIR /XD build /NFL /NDL /NJH /NJS /NP | Out-Null
  cmake -S $src -B $build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DGTA5OP_BUILD=$version" | Out-Null
  if ($LASTEXITCODE) { throw "cmake configure failed" }
  cmake --build $build
  if ($LASTEXITCODE) { throw "build failed" }
  $Core = "$build\gta5op_core.dll"
  Write-Host "built $version"
}

if ($Install) {
  $dir = Join-Path $GameDir "gta5op"
  $cores = Join-Path $dir "core"
  $backups = Join-Path $dir "backups"
  New-Item -ItemType Directory -Force $dir, $cores, "$dir\logs" | Out-Null
  $day = Get-Date -Format "yyyyMMdd"
  if ($Backup) { New-Item -ItemType Directory -Force $backups | Out-Null }

  # the loader: loaded by the game for as long as it runs, so it can only be replaced with the game closed. Every link
  # gives a different file, so whether it wants replacing is told by its sources, whose hash is kept beside it
  $asi = Join-Path $GameDir "gta5op.asi"
  $sources = "src\loader.cpp", "src\core_api.h", "src\shv.h", "ScriptHookV.def", "CMakeLists.txt" | ForEach-Object { (Get-FileHash (Join-Path $PSScriptRoot $_)).Hash }
  $loader = $sources -join " "
  $stamp = Join-Path $dir "loader_sources.txt"
  if ($built -and (-not (Test-Path $asi) -or -not (Test-Path $stamp) -or (Get-Content $stamp -Raw).Trim() -ne $loader)) {
    try {
      if ($Backup -and (Test-Path $asi)) { Copy-Item $asi "$backups\gta5op.asi.bak-pre-$Backup-$day" -Force }
      Copy-Item "$build\gta5op.asi" $asi -Force
      Set-Content $stamp $loader
      Write-Host "installed the loader gta5op.asi"
    } catch { Write-Warning "the loader's sources have changed and gta5op.asi is in use: the game keeps its old loader until it is closed and this is run again" }
  }

  $current = Get-ChildItem $cores -Filter "gta5op_core_*.dll" | Sort-Object Name | Select-Object -Last 1
  if ($Backup -and $current) { Copy-Item $current.FullName "$backups\gta5op_core.dll.bak-pre-$Backup-$day" -Force }
  # copied in under another name and renamed, so the loader never sees half a file
  $name = "gta5op_core_$version.dll"
  Copy-Item $Core "$cores\$name.tmp" -Force
  Rename-Item "$cores\$name.tmp" $name
  $newest = (Get-ChildItem $cores -Filter "gta5op_core_*.dll" | Sort-Object Name | Select-Object -Last 1).Name
  if ($newest -ne $name) { Write-Warning "$newest sorts after $name, so the loader keeps to it: remove it (or wait for its time to pass) for the new core to run" }
  # the newest three stay; one the game still has loaded or holds stays too, until a later install or the next game start
  Get-ChildItem $cores -Filter "gta5op_core_*.dll" | Sort-Object Name -Descending | Select-Object -Skip 3 | Where-Object Name -ne $name |
    ForEach-Object { try { Remove-Item $_.FullName -Force -ErrorAction Stop } catch {} }

  if (-not (Test-Path "$dir\gta5op.ini")) { Copy-Item "$PSScriptRoot\gta5op.ini" $dir }
  # openpilot's sounds, which the bridge has the plugin play as the car engages and hands back
  New-Item -ItemType Directory -Force "$dir\sounds" | Out-Null
  $pointers = @()
  foreach ($wav in Get-ChildItem "$PSScriptRoot\..\..\..\..\..\selfdrive\assets\sounds\*.wav") {
    $bytes = [IO.File]::ReadAllBytes($wav.FullName)
    if ($bytes.Length -lt 4 -or [Text.Encoding]::ASCII.GetString($bytes, 0, 4) -ne "RIFF") { $pointers += $wav.Name; continue }
    $to = Join-Path "$dir\sounds" $wav.Name
    if ((Test-Path $to) -and (Get-Item $to).Length -eq $bytes.Length) { continue }
    try { Copy-Item $wav.FullName $to -Force } catch { Write-Warning "$($wav.Name) is in use by the game and was not replaced" }
  }
  # a checkout without its Git LFS files has text pointers in their place, which play as silence
  if ($pointers) { Write-Warning "not installed, Git LFS pointers rather than WAVs: $($pointers -join ', '); fetch them with git lfs pull --include 'openpilot/selfdrive/assets/sounds/*'" }
  Write-Host "installed $name into $cores; the log is the newest file in $dir\logs"
}
