# Builds gta5op.asi and gta5op_core.dll with Visual Studio's CMake and Ninja, and optionally installs them into the game.
# The core reloads in a running game whenever it changes; gta5op.asi itself loads only at game start.
param(
  [string]$GameDir = "D:\SteamLibrary\steamapps\common\Grand Theft Auto V Enhanced",
  [switch]$Install
)
$ErrorActionPreference = "Stop"
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -prerelease -products * -property installationPath
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null

# MSVC and CMake handle \wsl.localhost sources poorly; build from a local copy
$src = Join-Path $env:LOCALAPPDATA "gta5op\src"
$build = Join-Path $env:LOCALAPPDATA "gta5op\build"
New-Item -ItemType Directory -Force $src, $build | Out-Null
robocopy $PSScriptRoot $src /MIR /XD build /NFL /NDL /NJH /NJS /NP | Out-Null
cmake -S $src -B $build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo | Out-Null
if ($LASTEXITCODE) { throw "cmake configure failed" }
cmake --build $build
if ($LASTEXITCODE) { throw "build failed" }

if ($Install) {
  $dir = Join-Path $GameDir "gta5op"
  New-Item -ItemType Directory -Force $dir | Out-Null
  try { Copy-Item "$build\gta5op.asi" $GameDir -Force }
  catch { Write-Warning "gta5op.asi is in use; it only changes at game start anyway" }
  Copy-Item "$build\gta5op_core.dll" $dir -Force
  if (-not (Test-Path "$dir\gta5op.ini")) { Copy-Item "$PSScriptRoot\gta5op.ini" $dir }
  Write-Host "installed into $GameDir"
}
