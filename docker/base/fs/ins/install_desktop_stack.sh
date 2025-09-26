#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

echo "====================DESKTOP STACK START===================="

apt-get install -y --no-install-recommends \
    xfce4 \
    xfce4-goodies \
    xfce4-terminal \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    dbus-x11 \
    xauth \
    xdg-utils \
    x11-apps \
    x11-xserver-utils \
    xdotool \
    wmctrl \
    imagemagick \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto-core \
    hunspell-en-us \
    libreoffice \
    libreoffice-gtk3 \
    libreoffice-help-en-us

# ensure novnc assets are reachable via index.html
if [ ! -e /usr/share/novnc/index.html ] && [ -e /usr/share/novnc/vnc.html ]; then
    ln -s /usr/share/novnc/vnc.html /usr/share/novnc/index.html
fi

mkdir -p /var/lib/x11vnc

# prepare runtime directories used by the supervised desktop services
mkdir -p /var/run/desktop-session/root
chmod 700 /var/run/desktop-session/root

echo "====================DESKTOP STACK END===================="
