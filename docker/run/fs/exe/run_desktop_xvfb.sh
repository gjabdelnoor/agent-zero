#!/bin/bash
set -euo pipefail

DISPLAY_NUM=${DESKTOP_DISPLAY:-:99}
LOCK_FILE="/tmp/.X${DISPLAY_NUM#:}-lock"

if [ -f "$LOCK_FILE" ]; then
    rm -f "$LOCK_FILE"
fi

exec Xvfb "$DISPLAY_NUM" -screen 0 ${DESKTOP_RESOLUTION:-1920x1080x24} -nolisten tcp -ac
