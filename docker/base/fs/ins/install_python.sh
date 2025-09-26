#!/bin/bash
set -e

echo "====================PYTHON START===================="

# Refresh package metadata before installing Python
apt-get update

echo "====================PYTHON 3.12 PACKAGES===================="
apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    python3-distutils

# Remove cached package data to keep the image small
rm -rf /var/lib/apt/lists/*

echo "====================PYTHON VERSION: $(python3 --version) ===================="
echo "====================PYTHON OTHERS: $(ls /usr/bin/python*) "

echo "====================PYTHON VENV===================="

# create and activate default venv
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# upgrade pip and install static packages
pip install --no-cache-dir --upgrade pip ipython requests
# Install some packages in specific variants
pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cpu

echo "====================PYTHON UV ===================="

curl -Ls https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# clean up pip cache
pip cache purge

echo "====================PYTHON END===================="
