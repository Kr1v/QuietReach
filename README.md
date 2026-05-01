# QuietReach

A passive domestic threat detection system. It listens to ambient sound, detects threat patterns using an on-device ML model, and silently sends an encrypted GPS alert to a trusted contact. No button press required. No recording sent. No human action needed.

---

## Why this exists

Around 70% of domestic violence victims report they couldn't safely use their phone during an incident. The abuser is right there. Opening an app, dialing a number, typing a message — any of these can escalate the situation.

QuietReach is built around that constraint. The device sits passively, does all detection locally, and triggers a silent alert automatically when threat patterns persist for 8+ seconds. The victim doesn't touch the phone. The trusted contact gets an SMS with location. That's it.

---

## How it works

```
Microphone → Feature Extraction → ML Classifier → Ensemble Scorer → Dispatcher → SMS Alert
                                                         ↑
                                              Vibration Sensor + Time Context
```

1. **Calibration** — on startup, listens to ambient noise for 10 seconds to establish a baseline. This is what makes it work across different environments (quiet apartment vs. noisy building).

2. **Detection loop** — extracts 43 audio features (MFCCs, spectral centroid, ZCR, RMS energy) from 3-second sliding windows, normalizes them against the baseline, runs inference.

3. **Ensemble scoring** — combines audio classifier confidence (60%), vibration/impact anomaly (25%), and time-of-day context (15%). Night hours score higher because incidents are more likely then.

4. **Dispatch logic** — threat score must stay above 0.72 for 8 consecutive seconds before anything triggers. A 3-second secondary confirmation window runs before the alert actually sends. Cooldown of 10 minutes between alerts prevents spam.

5. **Alert** — encrypted location payload, SMS via Twilio, optional Firebase push fallback.

Zero audio is ever saved or transmitted.

---

## Installation

**Prerequisites:**
- Python 3.10+
- PortAudio (for PyAudio): `brew install portaudio` (macOS) or `apt install portaudio19-dev` (Ubuntu)

```bash
git clone https://github.com/yourname/quietreach
cd quietreach

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your Twilio credentials and encryption key
```

**Generate encryption key:**
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
Paste the output into `ENCRYPTION_KEY` in your `.env`.

**Run:**
```bash
python main.py
```

Headless mode (no terminal UI):
```bash
python main.py --no-ui
```

List available microphone devices:
```bash
python main.py --list-devices
```

Use a specific mic (e.g. device index 2):
```bash
python main.py --device 2
```

---

## Training your own model

The included model was trained on ESC-50 environmental sounds. You can retrain it, add custom samples, or tune the threshold.

**Quick start:**
```bash
python -m model.trainer --epochs 30
```

This downloads ESC-50 automatically (~600MB), trains a Keras classifier, converts it to TFLite, and saves it to `model/saved/`.

**With custom labeled samples:**

Put your audio files in:
```
data/custom/threat/   ← raised voices, impacts, crying, anything threatening
data/custom/normal/   ← TV audio, conversation, ambient noise
```

Then retrain:
```bash
python -m model.trainer --epochs 50
```

**CLI options:**
```
--epochs N           Training epochs (default: 30)
--threshold FLOAT    Decision threshold for evaluation printout (default: 0.72)
--output-path PATH   Where to save the TFLite model
--sklearn-only       Skip Keras, train the GradientBoosting fallback only
--skip-download      Don't re-download ESC-50 if it's already in data/
```

---

## Privacy guarantee

This is the most important section.

**What QuietReach does NOT do:**
- Save audio to disk — never, under any condition
- Send audio to a server — the alert contains location + timestamp only
- Transcribe or describe audio content — the SMS has no audio information in it
- Use a cloud inference API — all ML runs on your device

**What it does:**
- Extract numerical features from audio windows (MFCCs, energy levels, etc.)
- Run those features through a local classifier
- Wipe the audio buffer from RAM immediately after feature extraction
- Send an encrypted location payload to your trusted contact when a threat is detected

The threat model here is: a device left running in a home, potentially examined afterward. Audio content should not be recoverable. The memory wiping in `privacy/memory_cleaner.py` is a best-effort mitigation — it's not cryptographically guaranteed under all OS conditions, but it's better than relying on Python's GC to do the right thing.

---

## Known limitations

**IP location accuracy** — the location in the alert is IP-based, which is accurate to about 1km in urban areas and much worse in rural ones. It's enough to identify the neighborhood, not the building. This is the biggest limitation for real-world use. v2 would use real GPS from a mobile app.

**Background noise sensitivity** — the calibration step helps significantly, but the system can still produce elevated scores in consistently noisy environments (busy street traffic, loud HVAC, etc.). Raising `THREAT_THRESHOLD` in `.env` is the main tuning knob. You might need values up to 0.82 in loud environments.

**Training data breadth** — the ESC-50 categories we use for "threat" audio are imperfect proxies. Glass breaking and crying register well. Raised voices are harder because they overlap heavily with normal speech. Custom labeled samples from your actual environment help a lot.

**No mobile GPS integration** — see above. The Python version is deliberately a validation prototype.

**Single trusted contact** — the current SMS implementation sends to one number. Adding multiple contacts is a one-line change in `alert/sms.py` but hasn't been tested.

---

## Running tests

```bash
python -m pytest tests/ -v
```

Tests don't require a microphone, trained model, Twilio credentials, or Firebase. They use synthetic audio and mocked external services.

---

## What v2 looks like

The Python version validates the core ML pipeline. Once that's solid, v2 is a Flutter mobile app that wraps the same logic:

- Real GPS instead of IP-based location
- On-device CoreML (iOS) / TFLite (Android) inference
- Multiple trusted contacts with contact list UI
- Community-labeled training data through an opt-in anonymized contribution mode
- Encrypted local audit log the user can review and delete
- Companion watch app for wrist-based detection

---

## Why Python first

The temptation with a project like this is to jump straight to mobile. But mobile adds ~3x the development friction: Xcode/Android Studio setup, platform API permissions, app store review, cross-platform audio handling.

Python lets you validate whether the ML pipeline actually works before betting on a mobile implementation. The feature extraction in `audio/processor.py`, the calibration approach in `audio/calibrator.py`, and the ensemble scoring in `model/threshold.py` — all of that needs to be right before worrying about SwiftUI layouts.

The answer turned out to be: calibration-based normalization is essential (raw features are inconsistent across environments), 8 seconds of consecutive high-score is about right for false-positive prevention, and the vibration proxy in demo mode is surprisingly effective as a secondary signal.

Now those answers are known. Mobile can start from a validated design.

---

## Acknowledgments

ESC-50 dataset: Karol J. Piczak, "ESC: Dataset for Environmental Sound Classification"

Crisis counseling input on time-of-day risk weighting: thanks to N. for the real-world context.
