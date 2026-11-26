mkdir -p /dev/shm/msgq_gta5  # msgq prefix dir: gone after a reboot, and the bridge starts before openpilot makes it
cd /home/amy/git/openpilot-slowroads && source .venv/bin/activate
export OPENPILOT_PREFIX=gta5 FINGERPRINT=TESLA_MODEL_3 GTA5_DEBUG=1 GTA5_MAP=/home/amy/gta5map_lanes GTA5_ROUTER=http://localhost:8002 GTA5_STOP_DIRECTION=1 GTA5_LOG=/home/amy/gta5test/bridge.jsonl GTA5_NAVTUNE=/home/amy/gta5test/navtune.json GTA5_NAV_MSGS=1 GTA5_DEBUG_OVERLAY=on GTA5_GPSROUTE=on 
exec script -aqfc './openpilot/tools/sim/run_bridge.py --simulator gta5' /home/amy/gta5test/bridge.log
