# Launch scripts (October capture setup)

The local launch scripts that ran openpilot, the GTA bridge and Valhalla, and recorded openpilot-format routes, as
they stood in `~/gta5test/` (October 2026). Kept on `keep/capture` beside the code they drove; see `../README.md`
("Running", "Drivers", "Recording", "Navigation") for how the pieces fit.

They hard-code the old paths (`/home/amy/git/openpilot-slowroads`, `/home/amy/gta5test`); change those before reuse.

| file | what it is |
|---|---|
| `svc.sh` | service script for openpilot, the bridge and Valhalla together |
| `run_bridge.sh` | starts the GTA bridge from the checkout's `.venv` |
| `run_openpilot.sh` | starts openpilot through `tools/sim/launch_openpilot.sh` with the nav model (`MODELD_PKL`), a model label and the UI window position file |
| `run_valhalla.sh` | starts the Valhalla routing server for navigation |
| `bridge_env` | environment variables for the bridge |
| `ui_env` | environment variables for the openpilot UI (window position file) |
| `ui_place.ps1` | Windows PowerShell: places the openpilot UI window on the desktop |
| `ui_window.json` | the window position `ui_place.ps1` uses |
| `windows` | plain-text notes kept with the scripts |
| `rec_map.sh` | a recording run over the map: a driver drives while `record_run.py` records (uses `gta5_cmd.py`, optional `GTA5_BLINDSPOT_TUNE` file) |
| `record.sh` | runs `record_run.py run` |
| `record_settings.txt` | the recorder settings used for the runs |

Not kept: the Windows host launcher `start_host.ps1` and the A/B harness `ftab_run2.sh` (dropped).
