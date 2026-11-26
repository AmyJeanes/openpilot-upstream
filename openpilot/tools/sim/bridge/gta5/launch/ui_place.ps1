# Saves or restores where openpilot's UI window (a WSLg window titled "UI") sits on the Windows desktop.
#   -Save          record the current rect in ui_window.json (next to this script)
#   -Restore       wait up to -WaitSeconds for the window, then move it to the saved rect
#   -List          print the UI windows found and their rects
#   -Handles       print the UI windows' handles, comma-separated
# -Exclude <handles> makes -Restore wait for a new window rather than move the old one, still closing during a restart.
param([switch]$Save, [switch]$Restore, [switch]$List, [switch]$Handles, [int]$WaitSeconds = 60, [string]$Exclude = "")

Add-Type @"
using System; using System.Runtime.InteropServices; using System.Text;
public class UiPlace {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc p, IntPtr l);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
  [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr c);
}
"@
# physical pixels on every monitor, so saved and restored rects agree across scaled displays
[void][UiPlace]::SetProcessDpiAwarenessContext([IntPtr](-4))  # PER_MONITOR_AWARE_V2

$file = Join-Path $PSScriptRoot "ui_window.json"

function Find-Ui {
  $script:found = @()
  [UiPlace]::EnumWindows({
    param($h, $l)
    $sb = New-Object Text.StringBuilder 256
    [void][UiPlace]::GetWindowText($h, $sb, 256)
    # WSLg titles its windows "<title> (<distro>)", prefixed with "[WARN:...]" in some modes
    if ([UiPlace]::IsWindowVisible($h) -and $sb.ToString() -match '^(\[[^\]]*\] )?UI \(') { $script:found += $h }
    $true
  }, [IntPtr]::Zero) | Out-Null
  $script:found
}

function Get-Rect($h) {
  $r = New-Object UiPlace+RECT
  [void][UiPlace]::GetWindowRect($h, [ref]$r)
  [ordered]@{ x = $r.Left; y = $r.Top; width = $r.Right - $r.Left; height = $r.Bottom - $r.Top }
}

if ($List) {
  foreach ($h in Find-Ui) { "{0}: {1}" -f $h, ((Get-Rect $h) | ConvertTo-Json -Compress) }
}

if ($Handles) { (@(Find-Ui) | ForEach-Object { [long]$_ }) -join ',' }

if ($Save) {
  $windows = @(Find-Ui)
  if ($windows.Count -ne 1) { Write-Error "expected one openpilot UI window, found $($windows.Count)"; exit 1 }
  $rect = Get-Rect $windows[0]
  $rect | ConvertTo-Json | Set-Content -Encoding ascii $file
  "saved the UI window at $($rect | ConvertTo-Json -Compress)"
}

if ($Restore) {
  if (-not (Test-Path $file)) { exit 0 }
  $rect = Get-Content -Raw $file | ConvertFrom-Json
  $old = @($Exclude -split ',' | Where-Object { $_ } | ForEach-Object { [IntPtr][long]$_ })
  $deadline = (Get-Date).AddSeconds($WaitSeconds)
  do {
    $windows = @(Find-Ui | Where-Object { $old -notcontains $_ })
    if ($windows.Count) { break }
    Start-Sleep -Milliseconds 500
  } while ((Get-Date) -lt $deadline)
  if (-not $windows.Count) { Write-Error "no new openpilot UI window within $WaitSeconds s"; exit 1 }
  Start-Sleep -Seconds 1  # let WSLg finish placing it before moving it
  foreach ($h in $windows) {
    # SWP_NOZORDER | SWP_NOACTIVATE, so the game keeps focus and pin_ui.ps1 keeps the z-order
    [void][UiPlace]::SetWindowPos($h, [IntPtr]::Zero, $rect.x, $rect.y, $rect.width, $rect.height, 0x15)  # SWP_NOSIZE: the UI sets its own size (SCALE)
    "moved the UI window to $((Get-Rect $h) | ConvertTo-Json -Compress)"
  }
}
