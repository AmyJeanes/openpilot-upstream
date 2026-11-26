# GTA V bridge

Drives GTA V Enhanced with openpilot. [README.md](README.md) is the user-facing guide (setup, running, controls,
commands); this file is what an agent needs beyond it. Work happens on branch `gta5` (tracks `fork/gta5` on
AmyJeanes/openpilot-upstream), checkout `~/gta5op/openpilot`. Only commit and push when Amy says so.

## What lives here

- `plugin/`: the ScriptHookV plugin in C++ (`src/`, `CMakeLists.txt`, `build.ps1`, default `gta5op.ini`). It renders
  openpilot's cameras, reads and drives the car, and sends the nav route and debug overlay.
- The bridge: `gta5_bridge.py` (the simulator for `run_bridge.py --simulator gta5`), `gta5_link.py`, `gta5_relay.py`,
  `gta5_host.py`, `gta5_navd.py` and the other `gta5_*.py` modules, with a `test_*.py` beside most. `gta5_cmd.py`
  sends debug commands to the running bridge.
- Capture and recorder: `gta5_record.py`, `record_run.py`, `watch_views.py`, `watch_route.py`. The `launch/` scripts
  (`record.sh`, `rec_map.sh`, `bridge_env`) live only on `keep/capture`.
- Old drivers awaiting clean-up: `gta5_driver.py`, `gta5_drivers.py`, `gta5_mapdrive.py`, `gta5_expert.py`,
  `gta5_brain.py`, `mapdrive_*.py`, `e2e.py`, `bsm_trial.py` and their tests. Do not extend them; check what still
  uses one before deleting it.

## What moved

- `map/` tools now live in AmyJeanes/gta5-map under `maptools/` (its AGENTS.md covers the map, services and data).
  The copies in `map/` here are frozen: change them there, not here.
- Pinned old work is on `keep/nav-ui-tizi` (nav UI on the comma 3X) and `keep/capture` (capture setup including
  `launch/`). They are never deleted or rebased. Anything else old is in the archive bundles, not on a branch.

## Plugin build and hot-swap

From PowerShell on Windows (Visual Studio with the C++ workload), through the WSL share:

```powershell
& '\wsl.localhost\Ubuntu\home\amy\gta5op\openpilot\openpilot\tools\sim\bridge\gta5\plugin\build.ps1' -Install -Label x
```

- `-Label` is a word (letters, digits, dots, hyphens) that goes into the core's file name, the log's init line and the
  state's `plugin.build`. `-Backup <word>` keeps the core being replaced; `-Core <file>` reinstalls an old core as
  the newest. The header of `build.ps1` is the reference.
- Each install writes a fresh-named core, `<game>\gta5op\core\gta5op_core_<build time>[_<Label>].dll`, never over an
  existing file. The loader (`gta5op.asi`) runs the newest by name and swaps to a newer one within a second, game
  running or not. The newest three are kept. `gta5op.asi` itself only changes with the game closed.
- One log per game start: the newest file in `<game>\gta5op\logs\` (`gta5op_<game start>.log`), shared by the loader
  and every core of that run; the newest twenty are kept. The bridge state's `plugin` names the running core, build
  and log.
- The build compiles a local copy of the source, with its cache in `%LOCALAPPDATA%\gta5op` (`src`, `build`), because
  MSVC and CMake handle `\wsl.localhost` badly.
- A "vswhere.exe is not recognized" line in the output is harmless; the build still runs.

## Do not retry

- Hardware breakpoints crash the game (anti-tamper). Use logging, not a debugger breakpoint.
- `GET_ENTITY_TYPE` / `GET_ENTITY_MODEL` on a map-object handle from a shape test crashes ScriptHookV. The fix is a
  pool check plus an SEH guard. Never reinstall the old `rays1` core.
- The plugin cannot write `gta5op.ini` while the game runs (the game holds the folder's files open), so
  `camera forward=` and similar are not saved until restart; likewise it cannot replace a sound file.
- F5-F8 are GTA's character switch, so openpilot uses Insert. The Home key freezes the game: never send it.
- The repo's relative `.venv/bin/git-lfs` filter leaves LFS pointer files in new worktrees (sounds then play as
  silence). Create worktrees with `-c filter.lfs.*` overrides, then `git lfs pull`.
- Rendering textures mid-frame fails on the 3X (nav UI).
