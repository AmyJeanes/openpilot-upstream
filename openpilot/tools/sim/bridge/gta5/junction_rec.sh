#!/bin/bash
# junction_rec.sh <plan.json> <name> [junction_run.py run options]: the junction recording run (junction_run.py) in tmux
# window gta5:jrec. After a stop a restart may get past (exit 3: the game gone, failures in a row, bridge restarts) it
# waits for the game's state to come again and resumes the same run, up to MAX_RESTARTS times (6) and RESUME_WAIT s
# (7200) of waiting each; it ends on the plan done, --until or --hours up, the stop file, or the driver taking over.
#   .../junction_rec.sh /mnt/e/juncrec/plan_jr1.json jr1 --until 2026-10-09T08:00
#   .../junction_rec.sh --dry-run /mnt/e/juncrec/plan_jr1.json jr1    the trips in order, checked on the live router
#   touch ~/gta5test/recruns/STOP      stop it after the trip being driven (the window stays open with the summary)
#   cat ~/gta5test/recruns/status.json; ~/gta5test/recruns/<name>.manifest.json, <name>.log
# Expert settings: ~/gta5test/record_settings.txt, read before each trip, as record.sh's. The bridge is restarted with
# BRIDGE_EXTRA (default: expert mode, recording to /mnt/e/gta5rec, the map overlay and GPS route off: the overlay's thread
# stalls the bridge's frames, and the plugin doesn't draw it on recordings anyway) unless it runs with all of it.
# openpilot runs alongside, as for record.sh (modeld's outputs and desire input go into gta5.npz; it stays disengaged).
REPO=${REPO:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
VENV=${VENV:-$HOME/git/openpilot-slowroads/.venv}  # a worktree has none of its own
T=~/gta5test
export BRIDGE_EXTRA="${BRIDGE_EXTRA:-GTA5_EXPERT=1 GTA5_RECORD=/mnt/e/gta5rec GTA5_DEBUG_OVERLAY=off GTA5_GPSROUTE=off}"
STOP=${STOP:-$T/recruns/STOP}
MAX_RESTARTS=${MAX_RESTARTS:-6}
RESUME_WAIT=${RESUME_WAIT:-7200}
py() { (cd "$REPO" && source "$VENV/bin/activate" && PYTHONPATH="$REPO" python -m openpilot.tools.sim.bridge.gta5.junction_run "$@"); }

if [ "$1" = --dry-run ]; then
  shift; plan=$1 name=$2; shift 2
  py run --plan "$plan" --name "$name" --dry-run "$@"; exit
fi

if [ "$1" = --inner ]; then
  shift; plan=$1 name=$2; shift 2
  until_arg=""; prev=""
  for a in "$@"; do [ "$prev" = --until ] && until_arg=$a; prev=$a; done
  first=--clear-stop n=0
  while :; do
    py run --plan "$plan" --name "$name" --fix-bridge --randomise $first "$@"
    code=$?; first=""
    [ $code -eq 3 ] || { echo "junction run ended (exit $code)"; break; }
    n=$((n + 1))
    [ $n -gt "$MAX_RESTARTS" ] && { echo "stopped $n times: not resuming again"; break; }
    echo "$(date +%T) the run stopped (exit 3): resuming once the game sends state again (restart $n of $MAX_RESTARTS)"
    waited=0
    while :; do
      [ -e "$STOP" ] && { echo "stop file: not resuming"; exit 0; }
      [ -n "$until_arg" ] && [ "$(date +%s)" -ge "$(date -d "$until_arg" +%s)" ] && { echo "--until passed"; exit 0; }
      age=$(( $(date +%s) - $(stat -c %Y $T/bridge.jsonl 2>/dev/null || echo 0) ))
      [ $age -lt 10 ] && break
      [ $waited -ge "$RESUME_WAIT" ] && { echo "no game state for ${RESUME_WAIT} s: not resuming"; exit 0; }
      sleep 30; waited=$((waited + 30))
    done
    sleep 20  # let the game settle
  done
  exit 0
fi

[ $# -ge 2 ] || { sed -n '2,13p' "$0"; exit 1; }
tmux has-session -t gta5 2>/dev/null || tmux new-session -d -s gta5 -n shell
if tmux list-windows -t gta5 -F '#W' | grep -qx jrec; then
  echo "a jrec window is open already (tmux attach -t gta5); close it first"; exit 1
fi
if tmux list-windows -t gta5 -F '#W' | grep -qx record; then
  echo "a record window (record.sh) is open: one recording run at a time"; exit 1
fi
tmux new-window -d -t gta5 -n jrec "BRIDGE_EXTRA='$BRIDGE_EXTRA' bash $(printf '%q ' "$0" --inner "$@"); echo; echo junction recording ended; read"
echo "started in tmux window gta5:jrec (tmux attach -t gta5; status: $T/recruns/status.json)"
