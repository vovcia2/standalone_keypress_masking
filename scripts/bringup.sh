#!/usr/bin/env bash
# keymask hardware bring-up for the AIY Voice HAT v1.
# Run ON THE PI, before any application code:   bash scripts/bringup.sh
# It never edits config files for you; it prints exactly what to put where.
set -u

CONFIG_TXT=/boot/firmware/config.txt
[ -f "$CONFIG_TXT" ] || CONFIG_TXT=/boot/config.txt
echo "== 1. Device-tree overlay ($CONFIG_TXT)"
if grep -q '^dtoverlay=googlevoicehat-soundcard' "$CONFIG_TXT"; then
  echo "   OK: dtoverlay=googlevoicehat-soundcard present"
else
  echo "   MISSING. Add this line to $CONFIG_TXT and reboot:"
  echo "       dtoverlay=googlevoicehat-soundcard"
fi
if grep -q '^dtparam=audio=on' "$CONFIG_TXT"; then
  echo "   WARNING: dtparam=audio=on is enabled (onboard bcm2835 audio). Either comment it"
  echo "   out so the HAT becomes card 0, or keep it and use the HAT's card index below."
fi

echo
echo "== 2. ALSA cards"
arecord -l 2>/dev/null | grep -iE 'googlevoi|voicehat' || echo "   capture card NOT found"
aplay   -l 2>/dev/null | grep -iE 'googlevoi|voicehat' || echo "   playback card NOT found"
CARD=$(arecord -l 2>/dev/null | grep -iE 'googlevoi|voicehat' | sed -n 's/^card \([0-9]*\):.*/\1/p' | head -1)
if [ -z "${CARD:-}" ]; then
  echo "   Cannot continue without the card. Fix step 1 and rerun."
  exit 1
fi
DEV="hw:${CARD},0"
echo "   -> use   audio.device: \"$DEV\"   in config.yaml"

echo
echo "== 3. Supported hw params (capture)"
# --dump-hw-params prints the card's capabilities and then fails on purpose
# (we ask for an impossible rate), so no audio is actually moved here.
CAP=$(arecord --dump-hw-params -D "$DEV" -d 1 -r 1 /dev/null 2>&1)
echo "$CAP" | grep -E 'FORMAT|RATE|CHANNELS|PERIOD_SIZE|BUFFER_SIZE' | sed 's/^/   /'
echo "== 3. Supported hw params (playback)"
PB=$(aplay --dump-hw-params -D "$DEV" -d 1 -r 1 -c 2 /dev/zero 2>&1)
echo "$PB" | grep -E 'FORMAT|RATE|CHANNELS|PERIOD_SIZE|BUFFER_SIZE' | sed 's/^/   /'

# Pick a format both directions support: S16_LE preferred, else S32_LE.
CAP_FMTS=$(echo "$CAP" | sed -n 's/^FORMAT: *//p')
PB_FMTS=$(echo "$PB" | sed -n 's/^FORMAT: *//p')
FMT=""
for f in S16_LE S32_LE S24_LE; do
  if echo "$CAP_FMTS" | grep -qw "$f" && echo "$PB_FMTS" | grep -qw "$f"; then FMT=$f; break; fi
done
if [ -z "$FMT" ]; then
  # fall back to whatever capture offers (the ICS-43432 mics are 32-bit I2S)
  FMT=$(echo "$CAP_FMTS" | awk '{print $1}')
  echo "   NOTE: no common format found; using capture format $FMT"
fi
case "$FMT" in
  S16_LE) DTYPE=int16 ;;
  S32_LE) DTYPE=int32 ;;
  *) echo "   format $FMT is not supported by keymask (needs S16_LE or S32_LE); STOP and report"; exit 3 ;;
esac
RATE=48000
if ! echo "$CAP" | grep -E '^RATE' | grep -q 48000; then
  echo "   WARNING: capture does not list 48000 Hz; check the RATE line above and set audio.rate"
fi
echo "   -> use   audio.rate: $RATE   audio.dtype: \"$DTYPE\"   ($FMT)   in config.yaml"

echo
echo "== 4. Full-duplex proof (5 s loopback: mic -> speaker, $FMT @ $RATE). Tap near the mic and listen."
echo "   If this fails with 'Device or resource busy' the driver cannot do simultaneous"
echo "   capture+playback on one card. STOP and report; do not work around it."
arecord -D "$DEV" -f "$FMT" -r "$RATE" -c 2 -d 5 2>/tmp/keymask_rec.err | aplay -D "$DEV" -f "$FMT" -r "$RATE" -c 2 2>/tmp/keymask_play.err
RC=$?
if [ $RC -eq 0 ] && ! grep -qi 'busy\|error' /tmp/keymask_rec.err /tmp/keymask_play.err; then
  echo "   OK: full duplex ran for 5 s"
else
  echo "   FAILED (rc=$RC):"; cat /tmp/keymask_rec.err /tmp/keymask_play.err | sed 's/^/   /'
  exit 2
fi

echo
echo "== 5. Python audio backend probe"
python3 - "$DEV" <<'EOF'
import sys
dev = sys.argv[1]
print("   python", sys.version.split()[0])
try:
    import sounddevice as sd
    print("   sounddevice", sd.__version__, "PortAudio", sd.get_portaudio_version()[1])
    idx = None
    for i, d in enumerate(sd.query_devices()):
        if dev.split(":")[1].split(",")[0] in d["name"] or "googlevoi" in d["name"].lower() or "voicehat" in d["name"].lower():
            idx = i; print("   PortAudio device", i, d["name"], "in", d["max_input_channels"], "out", d["max_output_channels"])
    if idx is None:
        print("   PortAudio does not list the HAT by name; keymask will fall back to pyalsaaudio")
except Exception as e:
    print("   sounddevice unavailable:", e)
try:
    import alsaaudio
    print("   pyalsaaudio OK; PCMs:", [p for p in alsaaudio.pcms() if 'hw' in p][:4])
except Exception as e:
    print("   pyalsaaudio unavailable:", e)
EOF

echo
echo "== Done. Put these in config.yaml:"
echo "     audio.device: \"$DEV\""
echo "     audio.rate: $RATE"
echo "     audio.dtype: \"$DTYPE\""
echo "   Next: python3 -m keymask duplex-test   (measures the smallest stable blocksize)"
