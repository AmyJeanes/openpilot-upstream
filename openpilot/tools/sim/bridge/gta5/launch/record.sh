#!/bin/bash
# record.sh [record_run.py run options]: the unattended AI expert recording run, in tmux window gta5:record
#   ~/gta5test/record.sh --hours 8                 start it (the bridge is restarted with BRIDGE_EXTRA if it lacks it)
#   ~/gta5test/record.sh --dry-run --n 30          only print the trips it would drive, here
#   touch ~/gta5test/recruns/STOP                  stop it (or Ctrl-C in the window: tmux attach -t gta5, window record)
#   cat ~/gta5test/recruns/status.json             how it's going; <name>.log, .jsonl, .summary.txt beside it
# Expert settings: ~/gta5test/record_settings.txt (read before each trip), or --settings "k=v ...".
# BRIDGE_EXTRA (default below) is what the bridge must run with, and is restarted with.
REPO=~/git/openpilot-slowroads
export BRIDGE_EXTRA="${BRIDGE_EXTRA:-GTA5_EXPERT=1 GTA5_RECORD=/mnt/e/gta5rec}"
run="cd $REPO && source .venv/bin/activate && python openpilot/tools/sim/bridge/gta5/record_run.py run"
case " $* " in
  *" --dry-run "*) eval "$run $(printf '%q ' "$@")"; exit;;
esac
tmux has-session -t gta5 2>/dev/null || tmux new-session -d -s gta5 -n shell
if tmux list-windows -t gta5 -F '#W' | grep -qx record; then
  echo "a record window is open already (tmux attach -t gta5); close it first"; exit 1
fi
# the window stays open afterwards, for the summary
tmux new-window -d -t gta5 -n record "BRIDGE_EXTRA='$BRIDGE_EXTRA' bash -c '$run --fix-bridge $(printf '%q ' "$@"); echo; echo record run ended; read'"
echo "started in tmux window gta5:record (tmux attach -t gta5; status: ~/gta5test/recruns/status.json)"
