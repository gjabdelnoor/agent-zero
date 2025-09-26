#!/bin/bash
set -euo pipefail

export DISPLAY=${DESKTOP_DISPLAY:-:99}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/var/run/desktop-session/root}
export HOME=${HOME:-/root}

mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# wait for X server to accept connections
for _ in $(seq 1 30); do
    if xset -display "$DISPLAY" q >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# ensure dbus daemon exits with session
touch "$HOME/.Xresources"

dbus_launch=("dbus-launch" "--exit-with-session")
if ! command -v dbus-launch >/dev/null 2>&1; then
    dbus_launch=()
fi

exec "${dbus_launch[@]}" startxfce4 --replace
