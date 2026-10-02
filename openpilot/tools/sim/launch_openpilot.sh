#!/usr/bin/env bash

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"
export SKIP_FW_QUERY="1"
export FINGERPRINT="${FINGERPRINT:-HONDA_CIVIC_2022}"  # or TESLA_MODEL_3, with the bridge given the same

export BLOCK="${BLOCK},camerad,loggerd,encoderd,micd,logmessaged,manage_athenad"
if [[ "$CI" ]]; then
  # TODO: offscreen UI should work
  export BLOCK="${BLOCK},ui"
fi

if [[ -n "$OPENPILOT_PREFIX" ]]; then
  # a prefix's messaging and params live in /dev/shm, which a restart (of WSL, say) clears
  python3 -c "import os; from openpilot.common.prefix import OpenpilotPrefix; OpenpilotPrefix(os.environ['OPENPILOT_PREFIX']).create_dirs()"
fi
python3 -c "from openpilot.selfdrive.test.helpers import set_params_enabled; set_params_enabled()"
if [[ "$FINGERPRINT" == TESLA* ]]; then
  # the simulated Tesla has openpilot longitudinal control, an alpha feature there
  python3 -c "from openpilot.common.params import Params; Params().put_bool('AlphaLongitudinalEnabled', True)"
fi

SCRIPT_DIR=$(dirname "$0")
OPENPILOT_DIR=$SCRIPT_DIR/../../

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$OPENPILOT_DIR/system/manager" && exec ./manager.py
