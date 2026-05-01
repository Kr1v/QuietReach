"""
setup_check.py — pre-flight environment checker

Run this before starting QuietReach for the first time.
It checks Python version, dependencies, microphone access,
.env config, and model availability.

    python setup_check.py

Prints a clear PASS / FAIL for each item.
"""

import sys
import os

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

errors = 0
warnings = 0


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    global errors, warnings
    if ok:
        print(f"{PASS}  {label}" + (f"  ({detail})" if detail else ""))
    elif warn_only:
        warnings += 1
        print(f"{WARN}  {label}" + (f"  — {detail}" if detail else ""))
    else:
        errors += 1
        print(f"{FAIL}  {label}" + (f"  — {detail}" if detail else ""))


# ── Python version ────────────────────────────────────────────────────────────
print("\n── Python ──────────────────────────────────────────────────────────────")
major, minor = sys.version_info[:2]
check(
    f"Python {major}.{minor}",
    major == 3 and minor >= 10,
    detail=sys.version.split()[0],
    warn_only=False if not (major == 3 and minor >= 10) else False,
)

# ── .env file ─────────────────────────────────────────────────────────────────
print("\n── Configuration ───────────────────────────────────────────────────────")
env_exists = os.path.exists(".env")
check(".env file exists", env_exists, detail="copy .env.example to .env and fill in values" if not env_exists else "")

if env_exists:
    from dotenv import load_dotenv
    load_dotenv()

    required_keys = ["TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM", "TRUSTED_NUMBER", "ENCRYPTION_KEY"]
    for key in required_keys:
        val = os.getenv(key, "")
        check(f"  {key} is set", bool(val), detail="missing in .env" if not val else "")

    # Check encryption key looks like a real Fernet key (44 chars base64)
    enc_key = os.getenv("ENCRYPTION_KEY", "")
    if enc_key:
        try:
            from cryptography.fernet import Fernet
            Fernet(enc_key.encode())
            check("  ENCRYPTION_KEY is valid Fernet key", True)
        except Exception:
            check("  ENCRYPTION_KEY is valid Fernet key", False,
                  detail='generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"')

# ── Core dependencies ─────────────────────────────────────────────────────────
print("\n── Dependencies ────────────────────────────────────────────────────────")

packages = [
    ("pyaudio",       "pyaudio",        False),
    ("librosa",       "librosa",        False),
    ("numpy",         "numpy",          False),
    ("sklearn",       "scikit-learn",   False),
    ("rich",          "rich",           False),
    ("dotenv",        "python-dotenv",  False),
    ("cryptography",  "cryptography",   False),
    ("geocoder",      "geocoder",       False),
    ("twilio",        "twilio",         False),
    ("flask",         "flask",          True),   # optional — only needed for phone sensor mode
    ("firebase_admin","firebase-admin", True),   # optional
    ("tensorflow",    "tensorflow",     True),   # optional — sklearn fallback exists
]

for import_name, pip_name, optional in packages:
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        check(f"{pip_name}", True, detail=version)
    except ImportError:
        check(
            f"{pip_name}",
            False,
            detail=f"pip install {pip_name}" + (" (optional)" if optional else ""),
            warn_only=optional,
        )

# ── Microphone access ─────────────────────────────────────────────────────────
print("\n── Microphone ──────────────────────────────────────────────────────────")
try:
    import pyaudio
    pa = pyaudio.PyAudio()
    device_count = pa.get_device_count()
    input_devices = [
        pa.get_device_info_by_index(i)
        for i in range(device_count)
        if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
    ]
    pa.terminate()
    check(f"Input devices found", len(input_devices) > 0,
          detail=f"{len(input_devices)} device(s) available" if input_devices else "no mic detected")
    if input_devices:
        default = input_devices[0]
        print(f"       Default: [{default['index']}] {default['name']}")
        print("       Run `python main.py --list-devices` to see all options")
except Exception as e:
    check("Microphone check", False, detail=str(e))

# ── Model files ───────────────────────────────────────────────────────────────
print("\n── Model ───────────────────────────────────────────────────────────────")
tflite_path = os.getenv("MODEL_PATH", "model/saved/quietreach_v1.tflite")
sklearn_path = os.getenv("SKLEARN_FALLBACK_PATH", "model/saved/quietreach_v1.pkl")

tflite_ok = os.path.exists(tflite_path)
sklearn_ok = os.path.exists(sklearn_path)

check("TFLite model", tflite_ok,
      detail=tflite_path if tflite_ok else f"not found at {tflite_path} — run: python -m model.trainer",
      warn_only=True)
check("sklearn fallback model", sklearn_ok,
      detail=sklearn_path if sklearn_ok else f"not found — run: python -m model.trainer --sklearn-only",
      warn_only=True)

if not tflite_ok and not sklearn_ok:
    print(f"\n  {WARN}  No model found. QuietReach will start with DummyClassifier (always returns 0.05).")
    print("       To train: python -m model.trainer --epochs 30")
    warnings += 1
elif not tflite_ok and sklearn_ok:
    print(f"\n  {PASS}  sklearn fallback will be used (no TFLite needed).")

# ── Training data (optional) ──────────────────────────────────────────────────
print("\n── Training Data (only needed to retrain) ──────────────────────────────")
esc50_ok = os.path.exists("data/esc50")
custom_ok = os.path.exists("data/custom")
check("ESC-50 dataset", esc50_ok,
      detail="auto-downloaded by model/trainer.py" if not esc50_ok else "found",
      warn_only=True)
check("Custom samples", custom_ok,
      detail="optional — add your own to data/custom/threat/ and data/custom/normal/",
      warn_only=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ─────────────────────────────────────────────────────────────")
if errors == 0 and warnings == 0:
    print("  All checks passed. Run: python main.py\n")
elif errors == 0:
    print(f"  {warnings} warning(s), 0 errors. You can run the app but some features may be limited.")
    print("  Run: python main.py\n")
else:
    print(f"  {errors} error(s) must be fixed before running QuietReach.")
    print("  See details above.\n")

sys.exit(0 if errors == 0 else 1)
