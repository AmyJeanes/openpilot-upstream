#!/bin/bash
# svc.sh start|stop|restart|status [openpilot|bridge|valhalla|all] - each runs in a window of tmux session 'gta5'
# svc.sh ui-save remembers where openpilot's UI window is; each openpilot (re)start puts it back there.
# BRIDGE_EXTRA="VAR=x ..." adds env to the bridge.
# MODEL=big|cd210|<compiled .pkl> picks openpilot's driving model; it's remembered in ~/gta5test/model for later calls.
# MAPDIR=<map folder> picks the map for Valhalla (the folder's valhalla.json) and the bridge (GTA5_MAP); remembered in
# ~/gta5test/map_dir. Default: the lane-tagged build ~/gta5map_lanes; ~/gta5map is the old map, kept as the fallback.
REPO=~/git/openpilot-slowroads
T=~/gta5test

# BRIDGE_HOST=<addr> is the address the game connects to (written for the plugin); remembered in ~/gta5test/bridge_host,
# 'auto' forgets it (the WSL IP). 127.0.0.1 uses WSL's localhost forwarding, for when Windows can't reach the WSL IP.
[ -n "$BRIDGE_HOST" ] && { [ "$BRIDGE_HOST" = auto ] && rm -f $T/bridge_host || echo "$BRIDGE_HOST" > $T/bridge_host; }
[ -s $T/bridge_host ] && BRIDGE_EXTRA="GTA5_BRIDGE_HOST=$(cat $T/bridge_host) $BRIDGE_EXTRA"
# bridge_env: env for normal (non-harness) bridge starts, e.g. the nav UI; anything in BRIDGE_EXTRA, which comes later, overrides it
[ -s $T/bridge_env ] && BRIDGE_EXTRA="$(cat $T/bridge_env) $BRIDGE_EXTRA"
[ -n "$MODEL" ] && echo "$MODEL" > $T/model
MODEL=$(cat $T/model 2>/dev/null || echo big)
[ -n "$MAPDIR" ] && echo "$MAPDIR" > $T/map_dir
MAPDIR=$(cat $T/map_dir 2>/dev/null || echo /home/amy/gta5map_lanes)
# desire_flags holds the GTA default bool params written as openpilot or the bridge starts, e.g. "TurnDesireRefresh=1 TurnDesireFlush=1 NavShowLanesAlways=1"
# (the turn desire ones from ftab11-13); the A/B harness writes its own per block and sets FTAB_PARAMS to keep these out.
desire_defaults() {
  if [ -n "$FTAB_PARAMS" ] || [ ! -s $T/desire_flags ]; then return; fi
  OPENPILOT_PREFIX=gta5 PYTHONPATH=$REPO/openpilot:$REPO $REPO/.venv/bin/python -c "
import sys
from openpilot.common.params import Params
p = Params()
for kv in sys.argv[1:]:
  k, v = kv.split('=')
  p.put_bool(k, v == '1', block=True)
" $(cat $T/desire_flags)
}
case "$MODEL" in
  big) MODEL_ENV="MODELD_BIG=1"; MODEL_LABEL=big;;
  cd210|small) MODEL_ENV="MODELD_BIG=0"; MODEL_LABEL=cd210;;
  *) MODEL_ENV="MODELD_BIG=0 MODELD_PKL=$MODEL"; MODEL_LABEL=$(basename "$MODEL" .pkl); MODEL_LABEL=${MODEL_LABEL%_big_driving_tinygrad};;
esac
# the UI shows this under the set speed
MODEL_ENV="$MODEL_ENV GTA5_MODEL_LABEL=$(printf %q "$MODEL_LABEL")"
# a file modeld_dump (temporary diagnostic) makes modeld record its inputs on request (modeld_dump.py) into ~/gta5test/dump
# ui_env holds extra env for openpilot UI, e.g. "BIG=1 SCALE=0.4" (the comma 3X layout, window scaled down)
[ -s $T/ui_env ] && MODEL_ENV="$MODEL_ENV $(cat $T/ui_env)"
[ -f $T/modeld_dump ] && MODEL_ENV="$MODEL_ENV MODELD_DUMP=$T/dump"
# RECORD_ROUTE=1 (this call only) has openpilot save drives as routes in ~/.commagta5/media/0/realdata (loggerd +
# encoderd, ~2.4 GB per minute; ~/gta5test/video/compact.sh shrinks it); ~/gta5test/video/make_clip.sh renders one. The next openpilot restart without it stops.
[ "$RECORD_ROUTE" = 1 ] && MODEL_ENV="$MODEL_ENV SIM_RECORD=1"

write_runners() {
  cat > $T/run_openpilot.sh <<EOF
cd $REPO && source .venv/bin/activate
export OPENPILOT_PREFIX=gta5 MODELD_DEV=CUDA $MODEL_ENV FINGERPRINT=TESLA_MODEL_3 GALLIUM_DRIVER=d3d12 BLOCK=soundd
exec script -qfc ./openpilot/tools/sim/launch_openpilot.sh $T/openpilot.log
EOF
  cat > $T/run_bridge.sh <<EOF
mkdir -p /dev/shm/msgq_gta5  # msgq prefix dir: gone after a reboot, and the bridge starts before openpilot makes it
cd $REPO && source .venv/bin/activate
export OPENPILOT_PREFIX=gta5 FINGERPRINT=TESLA_MODEL_3 GTA5_DEBUG=1 GTA5_MAP=$MAPDIR GTA5_ROUTER=http://localhost:8002 GTA5_STOP_DIRECTION=1 GTA5_LOG=$T/bridge.jsonl GTA5_NAVTUNE=$T/navtune.json $BRIDGE_EXTRA
exec script -aqfc './openpilot/tools/sim/run_bridge.py --simulator gta5' $T/bridge.log
EOF
  cat > $T/run_valhalla.sh <<EOF
cd $MAPDIR && . ~/gta5map/.venv/bin/activate
exec valhalla_service valhalla.json 2 > $MAPDIR/valhalla.log 2>&1
EOF
}

UI_PLACE="timeout 60 powershell.exe -NoProfile -ExecutionPolicy Bypass -File $(wslpath -w $T/ui_place.ps1)"

# unused: moving a WSLg window from Windows desyncs it (no clicks or drags); the UI restores its own position (UI_WINDOW_POS_FILE)
place_ui() {
  [ -s $T/ui_window.json ] || return 0
  local old=$($UI_PLACE -Handles 2>/dev/null | tr -d '\r')
  setsid nohup $UI_PLACE -Restore -WaitSeconds 90 -Exclude "$old" >> $T/ui_place.log 2>&1 < /dev/null &
}

stop_one() {
  case $1 in
    openpilot)
      pkill -INT -f '[s]ystem/manager/manager.py'
      for i in $(seq 40); do pgrep -f '[s]ystem/manager/manager.py' >/dev/null || break; sleep 0.5; done
      pkill -9 -f '[s]ystem/manager/manager.py'; pkill -f '[l]aunch_openpilot.sh';;
    bridge)
      pkill -INT -f '[r]un_bridge.py'
      for i in $(seq 20); do pgrep -f '[r]un_bridge.py' >/dev/null || break; sleep 0.5; done
      pkill -9 -f '[r]un_bridge.py';;
    valhalla) pkill -f '[v]alhalla_service';;
  esac
  tmux kill-window -t gta5:$1 2>/dev/null
}

start_one() {
  case $1 in openpilot|bridge) desire_defaults;; esac
  tmux has-session -t gta5 2>/dev/null || env -u MODEL -u BRIDGE_HOST -u MAPDIR -u RECORD_ROUTE tmux new-session -d -s gta5 -n shell  # keep svc.sh's one-off settings out of tmux's global env
  tmux new-window -d -t gta5 -n $1 "bash $T/run_$1.sh; sleep 5"
}

status() {
  for p in '[s]ystem/manager/manager.py' '[r]un_bridge.py' '[v]alhalla_service'; do
    printf '%-30s %s\n' "$p" "$(pgrep -f "$p" | wc -l)"
  done
}

write_runners
what=${2:-all}; [ "$what" = all ] && what='valhalla openpilot bridge'
for w in $what; do
  case $1 in
    start) start_one $w;;
    stop) stop_one $w;;
    restart) stop_one $w; sleep 1; start_one $w;;
  esac
done
[ "$1" = status ] && status
[ "$1" = ui-save ] && $UI_PLACE -Save
true
