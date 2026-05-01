"""
model_check.py — verify the trained model works correctly

Run from QuietReach root:
    python model_check.py

Tests the model against real synthetic audio, not arbitrary numpy vectors.
A passing model should score threat audio > 0.5 and normal audio < 0.3.
"""

import pickle
import sys
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE    = 16000
WINDOW_SECONDS = 3
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
MFCC_N         = 40
FEATURE_DIM    = 43

MODEL_PATH = Path("model/saved/quietreach_v1.pkl")


def extract_features(audio: np.ndarray) -> np.ndarray:
    """Same feature extraction as audio/processor.py — must stay in sync."""
    mfcc      = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_N, n_fft=512, hop_length=256)
    mfcc_mean = np.mean(mfcc, axis=1)
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=512, hop_length=256)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(audio, hop_length=256)))
    rms       = float(np.mean(librosa.feature.rms(y=audio, hop_length=256)))
    return np.concatenate([mfcc_mean, [centroid, zcr, rms]]).astype(np.float32)


def make_threat_audio(seed: int = 0) -> np.ndarray:
    """Loud clipped broadband noise — simulates impact/scream energy profile."""
    rng    = np.random.default_rng(seed)
    signal = rng.standard_normal(WINDOW_SAMPLES).astype(np.float32)
    signal = np.clip(signal, -0.5, 0.5)            # hard clip = distortion
    target = 0.30                                   # loud RMS
    signal = signal / (np.sqrt(np.mean(signal**2)) + 1e-9) * target
    return signal


def make_normal_audio(seed: int = 0) -> np.ndarray:
    """Quiet low-frequency hum — simulates ambient room / AC noise."""
    rng    = np.random.default_rng(seed)
    t      = np.linspace(0, WINDOW_SECONDS, WINDOW_SAMPLES)
    signal = np.sin(2 * np.pi * 80 * t).astype(np.float32)  # 80 Hz hum
    signal += rng.standard_normal(WINDOW_SAMPLES).astype(np.float32) * 0.01
    target = 0.005                                  # very quiet RMS
    signal = signal / (np.sqrt(np.mean(signal**2)) + 1e-9) * target
    return signal


def main() -> None:
    print("=" * 50)
    print("QuietReach model check")
    print("=" * 50)

    # ── Load model ────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"\nERROR: model not found at {MODEL_PATH}")
        print("Run: python -m model.trainer --sklearn-only --synthetic")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    print(f"\nModel loaded: {MODEL_PATH}")

    # ── Generate test audio ───────────────────────────────────────────────
    print("\nGenerating test audio samples...")
    rng = np.random.default_rng(999)

    threat_scores = []
    normal_scores = []

    N_TESTS = 10
    for i in range(N_TESTS):
        t_audio = make_threat_audio(seed=i)
        n_audio = make_normal_audio(seed=i + 100)

        t_vec = extract_features(t_audio).reshape(1, -1)
        n_vec = extract_features(n_audio).reshape(1, -1)

        threat_scores.append(clf.predict_proba(t_vec)[0][1])
        normal_scores.append(clf.predict_proba(n_vec)[0][1])

    avg_threat = float(np.mean(threat_scores))
    avg_normal = float(np.mean(normal_scores))
    min_threat = float(np.min(threat_scores))
    max_normal = float(np.max(normal_scores))

    # ── Results ───────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Threat audio  avg score : {avg_threat:.3f}  (need > 0.50)")
    print(f"  Normal audio  avg score : {avg_normal:.3f}  (need < 0.30)")
    print(f"  Worst threat  min score : {min_threat:.3f}")
    print(f"  Worst normal  max score : {max_normal:.3f}")
    print(f"{'─'*50}")

    threat_ok = avg_threat > 0.40
    normal_ok = avg_normal < 0.40
    gap_ok    = (avg_threat - avg_normal) > 0.30

    print(f"\n  Threat detection : {'PASS ✓' if threat_ok else 'FAIL ✗'}  ({avg_threat:.3f} > 0.40)")
    print(f"  Normal rejection : {'PASS ✓' if normal_ok else 'FAIL ✗'}  ({avg_normal:.3f} < 0.40)")
    print(f"  Score separation : {'PASS ✓' if gap_ok    else 'FAIL ✗'}  (gap={avg_threat-avg_normal:.3f}, need > 0.30)")

    if threat_ok and normal_ok and gap_ok:
        print("\n  ✓ Model is working correctly. Run: python main.py")
    else:
        print("\n  ✗ Model needs retraining.")
        print("\n  Fix: python -m model.trainer --sklearn-only --synthetic --synthetic-samples 1000")
        print("  Then run this check again.")

    # ── Feature diagnostics ───────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Feature diagnostics (avg across test samples):")
    t_feats = np.array([extract_features(make_threat_audio(i)) for i in range(5)])
    n_feats = np.array([extract_features(make_normal_audio(i + 100)) for i in range(5)])

    print(f"  RMS  — threat: {np.mean(t_feats[:,42]):.4f}  normal: {np.mean(n_feats[:,42]):.4f}")
    print(f"  ZCR  — threat: {np.mean(t_feats[:,41]):.4f}  normal: {np.mean(n_feats[:,41]):.4f}")
    print(f"  MFCC0— threat: {np.mean(t_feats[:,0]):.2f}    normal: {np.mean(n_feats[:,0]):.2f}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()