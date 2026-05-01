# Running QuietReach — Complete Setup Guide

---

## Prerequisites

| Requirement                     | Notes                                                         |
| ------------------------------- | ------------------------------------------------------------- |
| Python 3.10+                    | 3.11 recommended                                              |
| PortAudio                       | Required by PyAudio for mic access                            |
| A microphone                    | Built-in laptop mic works fine                                |
| Twilio account                  | Free trial covers ~200 SMS — [twilio.com](https://twilio.com) |
| (Optional) Google Cloud account | For Gemini API and Cloud Run                                  |

---

## Step 1 — Install system dependencies

**macOS:**

```bash
brew install portaudio
```

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install portaudio19-dev python3-dev ffmpeg
```

**Windows:**

```bash
# PyAudio has a pre-built wheel for Windows — no PortAudio install needed
pip install pipwin
pipwin install pyaudio
```

---

## Step 2 — Clone and set up virtual environment

```bash
git clone https://github.com/runtime-regret/quietreach
cd quietreach

python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

If PyAudio fails on Linux:

```bash
sudo apt install python3-pyaudio
# or
pip install pyaudio --global-option="build_ext" --global-option="-I/usr/include"
```

---

## Step 4 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_TOKEN=your_auth_token_here
TWILIO_FROM=+1xxxxxxxxxx
TRUSTED_NUMBER=+1xxxxxxxxxx
```

Generate your encryption key (run this once):

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the output into `ENCRYPTION_KEY=` in your `.env`.

---

## Step 5 — Run the pre-flight check

```bash
python setup_check.py
```

This verifies Python version, all dependencies, mic access, `.env` values, and model availability.
Fix anything marked with ✗ before proceeding.

---

## Step 6 — Train a model (first time only)

QuietReach won't detect anything meaningful without a trained model.
The trainer downloads the ESC-50 dataset automatically (~600 MB).

```bash
# Full training — TFLite + sklearn fallback (recommended)


# Fast option — sklearn only, no TensorFlow needed, trains in ~2 minutes
python -m model.trainer --sklearn-only

# With custom audio samples in data/custom/
python -m model.trainer --epochs 50
```

Trained models are saved to `model/saved/`. Training takes:

- `--sklearn-only`: ~2 minutes
- Full Keras + TFLite: ~15–30 minutes on CPU

---

## Step 7 — Run QuietReach

```bash
# Standard — with terminal dashboard
python main.py

# Headless — no UI, just logs (useful on servers / Raspberry Pi)
python main.py --no-ui

# List available microphone devices
python main.py --list-devices

# Use a specific microphone (e.g. USB mic at index 2)
python main.py --device 2
```

### What you'll see on startup:

```
09:14:02 [INFO]  Config validated OK.
09:14:02 [INFO]  Mic capture started (rate=16000, chunk=1024)
09:14:02 [INFO]  Calibrating ambient baseline (10s)...
  → Stay quiet for 10 seconds while it listens to the room
09:14:12 [INFO]  Calibration complete. Ambient RMS=0.0082 — environment looks good.
09:14:12 [INFO]  Detection loop started
```

The terminal dashboard then shows live threat level, noise level, and consecutive counter.

---

## Running tests

```bash
# All tests (no mic or model required)
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_audio.py -v
python -m pytest tests/test_classifier.py -v
python -m pytest tests/test_alert.py -v
```

---

## Getting sample audio files

The model trains on the **ESC-50** dataset. You can also use ESC-50 files to manually test the pipeline.

### Option A — Let the trainer download automatically

```bash
python -m model.trainer --epochs 1
# Downloads ESC-50 to data/esc50/ then stops after 1 epoch
# You can Ctrl+C once download is done
```

### Option B — Download ESC-50 manually

```bash
# ~600 MB zip
wget https://github.com/karoldvl/ESC-50/archive/master.zip -O esc50.zip
unzip esc50.zip
mv ESC-50-master data/esc50
```

ESC-50 audio files are in `data/esc50/audio/` as `.wav` files, 5 seconds each, 44.1 kHz.

### Good files to test with:

| File (in data/esc50/audio/)          | Expected behaviour              |
| ------------------------------------ | ------------------------------- |
| `1-100032-A-0.wav` (dog)             | Low score — normal animal sound |
| `1-17592-A-40.wav` (crying baby)     | Should raise threat score       |
| `1-100038-A-14.wav` (glass breaking) | Should raise threat score       |
| `4-157049-A-40.wav` (crying)         | Should raise threat score       |
| `1-21189-A-41.wav` (sneezing)        | Low/medium score                |
| `1-18527-A-28.wav` (clock tick)      | Low score — calm ambient        |

The exact filenames vary by ESC-50 version. Use the metadata CSV at `data/esc50/meta/esc50.csv` to find files by category:

```bash
grep "crying_baby" data/esc50/meta/esc50.csv | cut -d',' -f1 | head -5
```

### Option C — Free threat-relevant audio sources

These sites provide free audio under Creative Commons or public domain:

| Source            | URL                           | What to get                                                |
| ----------------- | ----------------------------- | ---------------------------------------------------------- |
| Freesound.org     | freesound.org                 | Search "glass breaking", "argument", "crying", "door slam" |
| BBC Sound Effects | bbcrewind.co.uk/sound-effects | High quality, free for personal/research use               |
| OpenSLR           | openslr.org                   | Speech datasets if you want voice-based threat detection   |
| DCASE dataset     | dcase.community               | Urban sound datasets including impact sounds               |

Download as `.wav` or `.mp3`, put in:

- `data/custom/threat/` — for sounds that should trigger alerts
- `data/custom/normal/` — for sounds that should NOT trigger alerts

Then retrain:

```bash
python -m model.trainer --epochs 50
```

### Option D — Record your own

```bash
# Generate synthetic test audio with Python (no download needed)
python - <<'EOF'
import numpy as np
import soundfile as sf
import os
os.makedirs("tests/sample_audio", exist_ok=True)

sr = 16000

# Silent clip — should produce ~0.05 threat score
silence = np.zeros(sr * 5, dtype=np.float32)
sf.write("tests/sample_audio/silence.wav", silence, sr)

# White noise burst — tests RMS spike detection
rng = np.random.default_rng(42)
noise = (rng.standard_normal(sr * 5) * 0.4).astype(np.float32)
sf.write("tests/sample_audio/noise_burst.wav", noise, sr)

print("Created: tests/sample_audio/silence.wav")
print("Created: tests/sample_audio/noise_burst.wav")
EOF
```

---

## Testing the full pipeline manually

Play a threat audio file through your system speakers while QuietReach is running:

**macOS:**

```bash
# In one terminal — start QuietReach
python main.py

# In another terminal — play the audio
afplay data/esc50/audio/1-17592-A-40.wav
```

**Linux:**

```bash
python main.py &
aplay data/esc50/audio/1-17592-A-40.wav
# or
ffplay -nodisp -autoexit data/esc50/audio/1-17592-A-40.wav
```

Watch the threat meter rise. If the score stays above `THREAT_THRESHOLD` (default 0.72) for 8+ seconds,
an SMS will fire to `TRUSTED_NUMBER`.

---

## Tuning for your environment

If you get too many false positives (alerts from TV, conversation, etc.):

```bash
# In .env, raise the threshold
THREAT_THRESHOLD=0.82
```

If genuine threats aren't triggering:

```bash
# Lower the threshold
THREAT_THRESHOLD=0.65
# Or reduce consecutive time required
CONSECUTIVE_SECONDS_REQUIRED=6
```

If the ambient noise warning appears on startup ("ambient level is high"):

```bash
# Calibrate in a quieter moment, or
# Raise threshold to compensate for noisy environment
THREAT_THRESHOLD=0.85
```

---

## Project structure recap

```
quietreach/
├── main.py               ← start here
├── setup_check.py        ← run before first launch
├── config.py             ← all settings
├── .env                  ← your secrets (never commit)
├── .env.example          ← template for .env
├── requirements.txt
│
├── audio/                ← mic capture, feature extraction, calibration
├── model/                ← classifier, threshold scorer, trainer
├── sensors/              ← vibration proxy
├── alert/                ← dispatcher, SMS, push, location
├── privacy/              ← encryption, memory wiping
├── ui/                   ← Rich terminal dashboard
└── tests/                ← unit tests (no mic/model needed)
```

---

## Common errors

**`OSError: [Errno -9996] Invalid input device`**
→ Run `python main.py --list-devices` and pass `--device N` with a valid index.

**`ModuleNotFoundError: No module named 'pyaudio'`**
→ On macOS: `brew install portaudio && pip install pyaudio`
→ On Linux: `sudo apt install portaudio19-dev && pip install pyaudio`

**`FileNotFoundError: No model found`**
→ Run `python -m model.trainer --sklearn-only` (fastest option).

**`ValueError: Invalid encryption key`**
→ Your `ENCRYPTION_KEY` in `.env` isn't a valid Fernet key.
Generate a new one: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`

**`TwilioRestException: Unable to create record`**
→ Check `TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_FROM` are correct.
On a free Twilio trial, `TRUSTED_NUMBER` must be a verified number in your Twilio console.
