#!/bin/bash
set -euo pipefail

VNC_HOST=${DESKTOP_VNC_HOST:-127.0.0.1}
VNC_PORT=${DESKTOP_VNC_PORT:-5901}
NOVNC_PORT=${DESKTOP_NOVNC_PORT:-6080}
NOVNC_WEB=${DESKTOP_NOVNC_WEB:-/usr/share/novnc/}
NOVNC_LISTEN=${DESKTOP_NOVNC_LISTEN:-127.0.0.1}

# ensure novnc assets exist
if [ ! -d "$NOVNC_WEB" ]; then
    echo "noVNC assets not found in $NOVNC_WEB" >&2
    exit 1
fi

exec websockify --web "$NOVNC_WEB" \
    --heartbeat 30 \
    "$NOVNC_LISTEN":"$NOVNC_PORT" \
    "$VNC_HOST":"$VNC_PORT"
