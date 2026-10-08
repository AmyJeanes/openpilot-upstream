# Keeps openpilot's UI window (a WSLg window titled "UI") and the route input viewer ("Route input") above the game. -Loop keeps pinning it, as when the UI restarts.
param([switch]$Loop)

# each bridge start launches a loop; one is enough
$fresh = $false
$mutex = New-Object System.Threading.Mutex($true, "gta5_pin_ui_loop", [ref]$fresh)
if ($Loop -and -not $fresh) { exit 0 }

Add-Type @"
using System; using System.Runtime.InteropServices; using System.Text;
public class PinUi {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc p, IntPtr l);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr h, int i);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
}
"@

function Pin {
  $script:windows = @()
  [PinUi]::EnumWindows({
    param($h, $l)
    $sb = New-Object Text.StringBuilder 256
    [void][PinUi]::GetWindowText($h, $sb, 256)
    # WSLg titles its windows "<title> (<distro>)", prefixed with "[WARN:...]" in some modes
    if ([PinUi]::IsWindowVisible($h) -and $sb.ToString() -match '^(\[[^\]]*\] )?(UI|Route input) \(') { $script:windows += $h }
    $true
  }, [IntPtr]::Zero) | Out-Null
  foreach ($h in $script:windows) {
    $topmost = ([PinUi]::GetWindowLong($h, -20) -band 0x8) -ne 0  # GWL_EXSTYLE, WS_EX_TOPMOST
    if (-not $topmost) {
      # HWND_TOPMOST, without moving, resizing or activating it
      [void][PinUi]::SetWindowPos($h, [IntPtr](-1), 0, 0, 0, 0, 0x13)
      Write-Output "pinned a window"
    }
  }
}

do {
  Pin
  if ($Loop) { Start-Sleep -Seconds 5 }
} while ($Loop)
