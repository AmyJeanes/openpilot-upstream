#!/bin/bash
# rec_map.sh [UNTIL=HH:MM, default 12:00] [PREFIX, default map1011]: the map driver's traffic-off recording
# (record_run.py run --settings driver=map --traffic off --randomise), runs named PREFIXa, PREFIXb, ... (the first free
# name), each until UNTIL. A run that ends early on its own (fail streak, bridge restarts, plugin wait) is followed by a
# respawn (gta5_cmd.py vehicle model3: a new car, or one at the player on foot after a death) and the next run.
#   start/resume: tmux new-window -d -t gta5 -n record "bash ~/gta5test/rec_map.sh 12:00; echo rec_map ended; read"
#   pause/stop:   touch ~/gta5test/recruns/STOP   (ends the trip being driven at once; the run writes its summary and
#                 this loop exits; the window then waits for Enter)
#   status:       cat ~/gta5test/recruns/status.json; tail ~/gta5test/recruns/<name>.log
REPO=~/git/openpilot-slowroads
R=~/gta5test/recruns
UNTIL=${1:-12:00}
PREFIX=${2:-map1011}
export BRIDGE_EXTRA="GTA5_EXPERT=1 GTA5_RECORD=/mnt/e/gta5rec GTA5_BLINDSPOT_TUNE=/home/amy/gta5test/bsm_tune.json"
cd $REPO && source .venv/bin/activate && export PYTHONPATH=$PWD
C="python openpilot/tools/sim/bridge/gta5/gta5_cmd.py"
rm -f $R/STOP  # starting this is resuming
i=0
for L in a b c d e f g h i j k l m n o p q r s t u v w x y z; do
  i=$((i + 1))
  name=$PREFIX$L
  if [ -e $R/$name.jsonl ] || [ -e $R/$name.log ]; then continue; fi
  [ -e $R/STOP ] && { echo "$(date +%T) stop file: stopping"; break; }
  end=$(date -d "today $UNTIL" +%s)
  left=$((end - $(date +%s)))
  [ $left -gt 360 ] || { echo "$(date +%T) past $UNTIL: done"; break; }
  hours=$(awk -v s=$left 'BEGIN { printf "%.3f", s / 3600 }')
  seed=$((1111 + 37 * i + 10#$(date +%H%M)))
  echo "$(date +%T) run $name seed $seed for $hours h"
  python openpilot/tools/sim/bridge/gta5/record_run.py run --name $name --seed $seed --hours $hours --settings driver=map \
    --traffic off --randomise --fix-bridge --bridge-extra "$BRIDGE_EXTRA" < /dev/null
  [ -e $R/STOP ] && { echo "$(date +%T) stop file: stopping"; break; }
  why=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("why", ""))' $R/status.json 2>/dev/null)
  echo "$(date +%T) run $name ended: $why"
  case "$why" in
    *" h up"*|*"engage key"*|*"took over"*|*"low disk"*|*SIGINT*|*SIGTERM*) break;;
  esac
  echo "$(date +%T) respawning the car, then the next run"
  timeout 60 $C vehicle model3 < /dev/null
  sleep 30
done
