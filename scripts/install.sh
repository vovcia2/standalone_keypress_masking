#!/usr/bin/env bash
# Install keymask as a systemd service on the Pi. Run from the repo root as root:
#   sudo bash scripts/install.sh
set -euo pipefail
DEST=/opt/keymask

apt-get install -y --no-install-recommends python3-numpy python3-yaml python3-gpiozero python3-alsaaudio libportaudio2 python3-pip
pip3 install --no-deps sounddevice 2>/dev/null || pip3 install --break-system-packages --no-deps sounddevice

id -u keymask >/dev/null 2>&1 || useradd --system --home "$DEST" --shell /usr/sbin/nologin -G audio,gpio keymask
mkdir -p "$DEST"
cp -r keymask tools config.yaml "$DEST"/
mkdir -p "$DEST/decoys"
if [ -d decoys ] && ls decoys/*.wav >/dev/null 2>&1; then cp decoys/*.wav "$DEST/decoys/"; fi
chown -R keymask:keymask "$DEST"
# config.yaml is rewritten by calibration, so the service user must own it
chmod 664 "$DEST/config.yaml"

cp systemd/keymask.service /etc/systemd/system/keymask.service
systemctl daemon-reload
systemctl enable keymask
echo "Installed to $DEST. Start with: sudo systemctl start keymask ; logs: journalctl -u keymask -f"
echo "If calibration.loop_latency_samples is null, the first start calibrates (keep the room quiet)."
