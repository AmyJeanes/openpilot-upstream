# Injects the Slow Roads bridge page script into the running game over its Chrome DevTools port.
# Runs on Windows PowerShell 5.1 so WSL can call it via interop without installing anything on Windows.
param(
  [Parameter(Mandatory = $true)][string]$ScriptDir,
  [int]$Port = 9339,
  [string]$Url = 'ws://localhost:8790',
  [string]$ExtraConfig = '{}'
)
$ErrorActionPreference = 'Stop'

try { $targets = Invoke-RestMethod "http://127.0.0.1:$Port/json/list" }
catch { throw "Slow Roads debug port $Port is not open. Add --remote-debugging-port=$Port to the game's Steam launch options." }
# the Steam (Electron) build loads from app://, the web version from slowroads.io
$page = $targets | Where-Object { $_.type -eq 'page' -and ($_.url -like 'app://*' -or $_.url -like '*slowroads.io*') } | Select-Object -First 1
if (-not $page) { throw "No Slow Roads page found on port $Port" }

$ws = New-Object System.Net.WebSockets.ClientWebSocket
$ct = [Threading.CancellationToken]::None
$ws.ConnectAsync([Uri]$page.webSocketDebuggerUrl, $ct).Wait()
$script:nextId = 0
$script:events = New-Object System.Collections.ArrayList
$buf = New-Object byte[] (4MB)

function Receive-Message {
  $ms = New-Object System.IO.MemoryStream
  do {
    $r = $ws.ReceiveAsync((New-Object ArraySegment[byte] -ArgumentList (, $buf)), $ct).Result
    $ms.Write($buf, 0, $r.Count)
  } while (-not $r.EndOfMessage)
  [Text.Encoding]::UTF8.GetString($ms.ToArray()) | ConvertFrom-Json
}

function Invoke-Cdp([string]$Method, $Params = @{}) {
  $id = ++$script:nextId
  $json = @{ id = $id; method = $Method; params = $Params } | ConvertTo-Json -Depth 10 -Compress
  $bytes = [Text.Encoding]::UTF8.GetBytes($json)
  $ws.SendAsync((New-Object ArraySegment[byte] -ArgumentList (, $bytes)), 'Text', $true, $ct).Wait()
  while ($true) {
    $msg = Receive-Message
    if ($msg.id -eq $id) {
      if ($msg.error) { throw "$Method failed: $($msg.error.message)" }
      return $msg.result
    }
    if ($msg.method) { [void]$script:events.Add($msg) }
  }
}

function Invoke-PageEval([string]$Expression) {
  $r = Invoke-Cdp 'Runtime.evaluate' @{ expression = $Expression; awaitPromise = $true; returnByValue = $true }
  if ($r.exceptionDetails) { throw "Page eval failed: $($r.exceptionDetails.exception.description)" }
  $r.result.value
}

# The game's state is module-scoped; a never-pausing conditional breakpoint just inside a method that runs every frame
# exposes it. $Pattern, if given, must match the method's opening code; its first group names the value to expose.
function Expose-Global([string]$Global, [string]$Marker, [string]$Value = 'this', [string]$Pattern = '') {
  if (Invoke-PageEval "!!window.$Global") { return }
  # enabling the debugger replays a scriptParsed event for every loaded script
  if (-not $script:debuggerOn) { [void](Invoke-Cdp 'Debugger.enable'); $script:debuggerOn = $true }
  $loc = $null
  foreach ($e in @($script:events | Where-Object { $_.method -eq 'Debugger.scriptParsed' -and $_.params.url -match '/_app/immutable/.*\.js$' })) {
    $src = (Invoke-Cdp 'Debugger.getScriptSource' @{ scriptId = $e.params.scriptId }).scriptSource
    $i = -1
    while (($i = $src.IndexOf($Marker, $i + 1)) -ge 0) {
      if (-not $Pattern) { break }
      $m = [regex]::Match($src.Substring($i, [Math]::Min(2000, $src.Length - $i)), $Pattern)
      if ($m.Success) { $Value = $m.Groups[1].Value; break }
    }
    if ($i -lt 0) { continue }
    $pre = $src.Substring(0, $i)
    $line = ([regex]::Matches($pre, "`n")).Count
    $loc = @{ scriptId = $e.params.scriptId; lineNumber = $line; columnNumber = $i - ($pre.LastIndexOf("`n") + 1) + $Marker.Length }
    break
  }
  if (-not $loc) { throw "Could not find '$Marker' in the game scripts; the game may have updated." }
  $bp = Invoke-Cdp 'Debugger.setBreakpoint' @{ location = $loc; condition = "(window.$Global=$Value,false)" }
  $deadline = (Get-Date).AddSeconds(10)
  while (-not (Invoke-PageEval "!!window.$Global")) {
    if ((Get-Date) -gt $deadline) { break }
    Start-Sleep -Milliseconds 100
  }
  [void](Invoke-Cdp 'Debugger.removeBreakpoint' @{ breakpointId = $bp.breakpointId })
  if (-not (Invoke-PageEval "!!window.$Global")) { throw "Game loop did not run ($Marker); is the game paused in a menu or minimized?" }
}

$script:debuggerOn = $false
Expose-Global '__srGame' 'renderLive(){'
# the driver's raw controls; the vehicle ignores them while the game's autodrive drives
Expose-Global '__srInput' 'handleInput(' -Pattern '^handleInput\(\w+\)\{[^}]{0,1200}?([\w$]+)\.signal\.Forward'
if ($script:debuggerOn) { [void](Invoke-Cdp 'Debugger.disable') }

$config = $ExtraConfig | ConvertFrom-Json
$config | Add-Member -NotePropertyName url -NotePropertyValue $Url -Force
[void](Invoke-PageEval ("window.__srbConfig = " + ($config | ConvertTo-Json -Compress) + "; true"))
# one function scope: the parts define factories that sr-page.js, last, puts together
$parts = 'lens.js', 'scene.js', 'controls.js', 'sr-page.js' | ForEach-Object { [IO.File]::ReadAllText((Join-Path $ScriptDir $_)) }
Invoke-PageEval ("(() => {`n" + ($parts -join "`n") + "`n})()")
$ws.CloseAsync('NormalClosure', '', $ct).Wait()
