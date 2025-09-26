#!/bin/bash
set -euo pipefail

export DISPLAY=${DESKTOP_DISPLAY:-:99}

VNC_PORT=${DESKTOP_VNC_PORT:-5901}
PASSWORD_FILE=${DESKTOP_VNC_PASSWORD_FILE:-/var/run/desktop-session/vnc.pass}
PASSWORD_VALUE=${DESKTOP_VNC_PASSWORD:-}
ALLOW_REMOTE=${DESKTOP_VNC_ALLOW_REMOTE:-false}

mkdir -p "$(dirname "$PASSWORD_FILE")"

if [ ! -f "$PASSWORD_FILE" ] || [ ! -s "$PASSWORD_FILE" ]; then
    if [ -n "$PASSWORD_VALUE" ]; then
        x11vnc -storepasswd "$PASSWORD_VALUE" "$PASSWORD_FILE" >/dev/null
    else
        PASSWORD_VALUE=$(python3 - <<'PY'
import secrets
alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
print(''.join(secrets.choice(alphabet) for _ in range(24)))
PY
)
        x11vnc -storepasswd "$PASSWORD_VALUE" "$PASSWORD_FILE" >/dev/null
        echo "[desktop_vnc] Generated random password. Inspect $PASSWORD_FILE to retrieve it." >&2
    fi
fi

chmod 600 "$PASSWORD_FILE"

# wait for desktop session before starting vnc bridge
for _ in $(seq 1 30); do
    if xset -display "$DISPLAY" q >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

ARGS=(
    x11vnc
    -display "$DISPLAY"
    -rfbport "$VNC_PORT"
    -shared \
    -forever \
    -quiet \
    -xkb \
    -ncache 10 \
    -ncache_cr \
    -rfbauth "$PASSWORD_FILE"
)

if [ "${ALLOW_REMOTE,,}" != "true" ]; then
    ARGS+=( -localhost )
fi

exec "${ARGS[@]}"
