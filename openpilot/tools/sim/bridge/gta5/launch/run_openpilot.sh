cd /home/amy/git/openpilot-slowroads && source .venv/bin/activate
export OPENPILOT_PREFIX=gta5 MODELD_DEV=CUDA MODELD_BIG=0 MODELD_PKL=/home/amy/git/openpilot-slowroads/openpilot/selfdrive/modeld/models/local_big_nav_n16_tinygrad.pkl GTA5_MODEL_LABEL=local_big_nav_n16_tinygrad BIG=1 SCALE=0.4 GTA5_WIDE_CAM=1 FPS=20 UI_WINDOW_POS_FILE=/home/amy/gta5test/ui_window_pos FINGERPRINT=TESLA_MODEL_3 GALLIUM_DRIVER=d3d12 BLOCK=soundd
exec script -qfc ./openpilot/tools/sim/launch_openpilot.sh /home/amy/gta5test/openpilot.log
