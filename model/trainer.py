"""
model/trainer.py — training script for QuietReach classifier

Standalone script — NOT imported by main.py.
Run from the QuietReach root directory:

    # Fastest — no TensorFlow, uses real ESC-50 data
    python -m model.trainer --sklearn-only --skip-download

    # Full — TFLite + sklearn
    python -m model.trainer --epochs 30 --skip-download

    # Offline — no data needed at all
    python -m model.trainer --sklearn-only --synthetic
"""

import argparse
import csv
import os
import pickle
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

# Import TF first on Windows — importing after librosa/sklearn causes DLL deadlock
print("Loading ML dependencies...", flush=True)
print("  tensorflow...", end=" ", flush=True)
HAS_TF = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
    print(f"OK ({tf.__version__})", flush=True)
except ImportError:
    print("not installed — sklearn only", flush=True)
except Exception as e:
    print(f"failed ({e}) — sklearn only", flush=True)

import numpy as np
import librosa
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

print("  numpy   OK", flush=True)
print("  librosa OK", flush=True)
print("  sklearn OK", flush=True)
print("Ready.\n", flush=True)

SAMPLE_RATE    = 16000
WINDOW_SECONDS = 3
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
MFCC_N         = 40
FEATURE_DIM    = 43

ESC50_URL  = "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip"
ESC50_DIR  = Path("data/esc50")
CUSTOM_DIR = Path("data/custom")

THREAT_CATEGORIES = {
    # Genuinely relevant to domestic incidents
    "crying_baby",        # infant/child distress
    "glass_breaking",     # impact / violence
    "door_wood_knock",    # aggressive knocking
    "door_wood_creaks",   # door sounds during incident
}

NORMAL_CATEGORIES = {
    # Clearly non-threatening ambient sounds
    "clock_tick",
    "keyboard_typing",
    "mouse_click",
    "water_drops",
    "rain",
    "wind",
    "crickets",
    "frog",
    "crow",
    "rooster",
    "sea_waves",
    "cat",
    "dog",
}


def download_esc50() -> Path:
    if ESC50_DIR.exists():
        wavs = list((ESC50_DIR / "audio").glob("*.wav")) if (ESC50_DIR / "audio").exists() else []
        if wavs:
            print(f"ESC-50 already at {ESC50_DIR} ({len(wavs)} files) — skipping.", flush=True)
            return ESC50_DIR
        import shutil
        shutil.rmtree(ESC50_DIR)

    print("Downloading ESC-50 (~600 MB)...", flush=True)
    zip_path = Path("data/esc50_raw.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    def _progress(blk, blk_sz, total):
        if total > 0:
            print(f"  {min(blk*blk_sz/total*100,100):5.1f}%  {blk*blk_sz/1_048_576:.0f} MB", end="\r", flush=True)

    try:
        urllib.request.urlretrieve(ESC50_URL, zip_path, reporthook=_progress)
        print("  100%  done.           ", flush=True)
    except Exception as exc:
        print(f"\nDownload failed: {exc}", flush=True)
        sys.exit(1)

    print("Extracting...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("data/")

    for name in ("ESC-50-master", "ESC-50-main"):
        p = Path("data") / name
        if p.exists():
            p.rename(ESC50_DIR)
            break
    else:
        candidates = [d for d in Path("data").iterdir() if d.is_dir() and "ESC" in d.name]
        if candidates:
            candidates[0].rename(ESC50_DIR)
        else:
            print("ERROR: extracted ESC-50 folder not found in data/", flush=True)
            sys.exit(1)

    zip_path.unlink()
    n = len(list((ESC50_DIR / "audio").glob("*.wav")))
    print(f"ESC-50 ready: {n} wav files at {ESC50_DIR}", flush=True)
    return ESC50_DIR


def load_esc50_samples(esc50_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    meta_path = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"

    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.", flush=True)
        sys.exit(1)
    if not audio_dir.exists():
        print(f"ERROR: {audio_dir} not found.", flush=True)
        sys.exit(1)

    audios:  list[np.ndarray] = []
    labels:  list[int]        = []
    skipped: int               = 0

    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        print(f"  CSV columns: {cols}", flush=True)

        for raw in reader:
            row      = {k.strip(): v.strip() for k, v in raw.items()}
            category = row.get("category", "")

            if category not in THREAT_CATEGORIES and category not in NORMAL_CATEGORIES:
                continue

            label = 1 if category in THREAT_CATEGORIES else 0
            fpath = audio_dir / row.get("filename", "")

            if not fpath.exists():
                skipped += 1
                continue

            try:
                audio, _ = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True, duration=WINDOW_SECONDS)
                if len(audio) < WINDOW_SAMPLES:
                    audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
                audios.append(audio[:WINDOW_SAMPLES])
                labels.append(label)
            except Exception as exc:
                print(f"  skipping {fpath.name}: {exc}", flush=True)
                skipped += 1

    n_threat = sum(labels)
    n_normal = len(labels) - n_threat
    print(f"ESC-50: {len(audios)} samples  ({n_threat} threat / {n_normal} normal)", flush=True)
    if skipped:
        print(f"  ({skipped} skipped)", flush=True)

    if len(audios) == 0:
        print("ERROR: 0 samples loaded. Check category names in esc50.csv match:", flush=True)
        print(f"  Threat: {sorted(THREAT_CATEGORIES)}", flush=True)
        print(f"  Normal: {sorted(NORMAL_CATEGORIES)}", flush=True)
        sys.exit(1)

    return audios, labels


def load_custom_samples() -> tuple[list[np.ndarray], list[int]]:
    if not CUSTOM_DIR.exists():
        print("No data/custom/ — using ESC-50 only.", flush=True)
        return [], []

    audios: list[np.ndarray] = []
    labels: list[int]        = []

    for name, val in [("threat", 1), ("normal", 0)]:
        d = CUSTOM_DIR / name
        if not d.exists():
            continue
        for fpath in list(d.glob("*.wav")) + list(d.glob("*.mp3")):
            try:
                audio, _ = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True, duration=WINDOW_SECONDS)
                if len(audio) < WINDOW_SAMPLES:
                    audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
                audios.append(audio[:WINDOW_SAMPLES])
                labels.append(val)
            except Exception as exc:
                print(f"  skipping {fpath.name}: {exc}", flush=True)

    n_threat = sum(labels)
    print(f"Custom: {len(audios)} samples  ({n_threat} threat / {len(labels)-n_threat} normal)", flush=True)
    return audios, labels


def generate_synthetic_samples(n_per_class: int = 400, seed: int = 42) -> tuple[list[np.ndarray], list[int]]:
    """
    Generate synthetic audio with genuinely different acoustic profiles.

    Threat  = clipped white noise bursts at high amplitude
              → high RMS, high ZCR, flat/noisy MFCC profile
    Normal  = pure low-frequency sine tones at low amplitude
              → low RMS, low ZCR, harmonic MFCC profile

    These two profiles are acoustically far apart in feature space,
    giving the classifier clean separable training signal.
    """
    print(f"Generating {n_per_class} threat + {n_per_class} normal synthetic samples...", flush=True)
    rng    = np.random.default_rng(seed)
    audios: list[np.ndarray] = []
    labels: list[int]        = []

    for _ in range(n_per_class):
        # THREAT: clipped broadband noise burst — high RMS, high ZCR, chaotic
        signal = rng.standard_normal(WINDOW_SAMPLES).astype(np.float32)
        # hard clip to simulate distortion/impact
        signal = np.clip(signal, -0.5, 0.5)
        # random amplitude envelope — some bursts, some sustained
        if rng.random() > 0.5:
            # burst: loud start, fade
            t   = np.linspace(0, 1, WINDOW_SAMPLES)
            env = np.exp(-t * rng.uniform(1.0, 4.0)) + 0.3
            signal = signal * env
        # target RMS 0.15-0.40 (loud)
        target_rms = rng.uniform(0.15, 0.40)
        current_rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
        signal = (signal / current_rms * target_rms).astype(np.float32)
        audios.append(signal)
        labels.append(1)

    for _ in range(n_per_class):
        # NORMAL: pure low-frequency tone — low RMS, low ZCR, harmonic
        t  = np.linspace(0, WINDOW_SECONDS, WINDOW_SAMPLES)
        f0 = rng.uniform(60, 300)   # low fundamental — room noise / hum / AC
        signal = np.sin(2 * np.pi * f0 * t).astype(np.float32)
        # add tiny amount of noise so features aren't perfectly clean
        signal += rng.standard_normal(WINDOW_SAMPLES).astype(np.float32) * 0.02
        # target RMS 0.002-0.015 (quiet)
        target_rms = rng.uniform(0.002, 0.015)
        current_rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
        signal = (signal / current_rms * target_rms).astype(np.float32)
        audios.append(signal)
        labels.append(0)

    print(f"Synthetic ready: {n_per_class} threat / {n_per_class} normal", flush=True)
    return audios, labels


def extract_features(audio: np.ndarray) -> np.ndarray:
    mfcc      = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_N, n_fft=512, hop_length=256)
    mfcc_mean = np.mean(mfcc, axis=1)
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=512, hop_length=256)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(audio, hop_length=256)))
    rms       = float(np.mean(librosa.feature.rms(y=audio, hop_length=256)))
    vec       = np.concatenate([mfcc_mean, [centroid, zcr, rms]]).astype(np.float32)
    assert vec.shape == (FEATURE_DIM,), f"Feature dim mismatch: {vec.shape}"
    return vec


def build_feature_matrix(audios: list[np.ndarray], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    n = len(audios)
    print(f"Extracting features from {n} samples...", flush=True)
    X, y = [], []
    for i, (audio, label) in enumerate(zip(audios, labels)):
        if i % 50 == 0:
            print(f"  {i}/{n}", flush=True)
        X.append(extract_features(audio))
        y.append(label)
    print(f"  {n}/{n} done.", flush=True)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_sklearn(X_train: np.ndarray, y_train: np.ndarray) -> sklearn.ensemble.GradientBoostingClassifier:
    print("Training GradientBoosting classifier...", flush=True)
    clf = sklearn.ensemble.GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42, verbose=1,
    )
    clf.fit(X_train, y_train)
    print("Training complete.", flush=True)
    return clf


def build_keras_model(input_dim: int):
    import tensorflow as tf
    # BatchNormalization removed — causes LLVM error during TFLite export
    # with Keras 3 on Windows. Dropout provides sufficient regularisation.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_keras(X_train, y_train, X_val, y_val, epochs):
    import tensorflow as tf
    print(f"Training Keras model ({epochs} epochs)...", flush=True)
    model    = build_keras_model(FEATURE_DIM)
    n_normal = int(np.sum(y_train == 0))
    n_threat = int(np.sum(y_train == 1))
    total    = n_normal + n_threat
    cw       = {0: total / (2 * n_normal + 1e-9), 1: total / (2 * n_threat + 1e-9)}
    print(f"Class weights: normal={cw[0]:.2f}  threat={cw[1]:.2f}", flush=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=32, class_weight=cw, verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        ],
    )
    return model


def export_sklearn_pickle(clf, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"sklearn model  →  {output_path}", flush=True)


def export_tflite(keras_model, output_path: Path) -> None:
    import tensorflow as tf
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(output_path, "wb") as f:
        f.write(converter.convert())
    print(f"TFLite model   →  {output_path}  ({output_path.stat().st_size/1024:.1f} KB)", flush=True)


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float, label: str) -> None:
    print(f"\n── {label} evaluation (threshold={threshold}) ──────────────", flush=True)
    probs = model.predict(X_test, verbose=0).flatten() if label == "keras" else model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(f"  Accuracy  : {sklearn.metrics.accuracy_score(y_test, preds):.3f}", flush=True)
    print(f"  Precision : {sklearn.metrics.precision_score(y_test, preds, zero_division=0):.3f}", flush=True)
    print(f"  Recall    : {sklearn.metrics.recall_score(y_test, preds, zero_division=0):.3f}", flush=True)
    print(f"  F1        : {sklearn.metrics.f1_score(y_test, preds, zero_division=0):.3f}", flush=True)
    print(f"  ROC-AUC   : {sklearn.metrics.roc_auc_score(y_test, probs):.3f}", flush=True)
    print("", flush=True)
    print(sklearn.metrics.classification_report(y_test, preds, target_names=["normal", "threat"]), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QuietReach classifier", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--threshold",         type=float, default=0.72)
    p.add_argument("--output-path",       type=str,   default="model/saved/quietreach_v1.tflite")
    p.add_argument("--sklearn-output",    type=str,   default="model/saved/quietreach_v1.pkl")
    p.add_argument("--skip-download",     action="store_true", help="Skip ESC-50 download — data already in data/esc50/")
    p.add_argument("--sklearn-only",      action="store_true", help="Train sklearn only — no TensorFlow needed")
    p.add_argument("--synthetic",         action="store_true", help="Use synthetic data — no files needed")
    p.add_argument("--synthetic-samples", type=int,   default=400, help="Samples per class for --synthetic")
    p.add_argument("--test-split",        type=float, default=0.15)
    p.add_argument("--val-split",         type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    t_start = time.time()

    if args.sklearn_only:
        print("Mode: sklearn-only\n", flush=True)
    elif HAS_TF:
        print("Mode: sklearn + TFLite\n", flush=True)
    else:
        print("Mode: sklearn-only (TensorFlow unavailable)\n", flush=True)

    # ── Data ─────────────────────────────────────────────────────────────
    if args.synthetic:
        print("Using synthetic data (--synthetic)\n", flush=True)
        all_audios, all_labels = generate_synthetic_samples(n_per_class=args.synthetic_samples)
    else:
        esc50_dir = ESC50_DIR if args.skip_download else download_esc50()
        if not esc50_dir.exists():
            print(f"ERROR: {esc50_dir} not found. Run without --skip-download or use --synthetic.", flush=True)
            sys.exit(1)
        esc_audios, esc_labels     = load_esc50_samples(esc50_dir)
        custom_audios, custom_labels = load_custom_samples()
        all_audios = esc_audios + custom_audios
        all_labels = esc_labels + custom_labels

    if len(all_audios) == 0:
        print("ERROR: 0 samples. Use --synthetic for offline training.", flush=True)
        sys.exit(1)
    if len(set(all_labels)) < 2:
        print(f"ERROR: only class {set(all_labels)} found — need both 0 and 1.", flush=True)
        sys.exit(1)

    # ── Features ──────────────────────────────────────────────────────────
    X, y = build_feature_matrix(all_audios, all_labels)
    print(f"\nDataset : {X.shape[0]} samples, {X.shape[1]} features", flush=True)
    print(f"Balance : {int(np.sum(y==1))} threat  /  {int(np.sum(y==0))} normal\n", flush=True)

    # ── Split ─────────────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=args.test_split, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X_temp, y_temp, test_size=args.val_split / (1.0 - args.test_split), stratify=y_temp, random_state=42
    )
    print(f"Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test\n", flush=True)

    # ── Normalise ─────────────────────────────────────────────────────────
    scaler    = sklearn.preprocessing.StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── Train sklearn ─────────────────────────────────────────────────────
    clf = train_sklearn(X_train_s, y_train)
    evaluate(clf, X_test_s, y_test, args.threshold, "sklearn")
    export_sklearn_pickle(clf, Path(args.sklearn_output))

    # ── Train Keras / TFLite ──────────────────────────────────────────────
    # TFLite export disabled — LLVM error on TF 2.16 + Keras 3 + Windows.
    # sklearn model above is fully functional. To re-enable TFLite, downgrade:
    #   pip install tensorflow==2.15.0 tf-keras==2.15.0
    if args.sklearn_only:
        print("Skipping Keras training (--sklearn-only)", flush=True)
    else:
        print("Skipping TFLite export (incompatible on TF 2.16 + Keras 3 + Windows)", flush=True)
        print("sklearn model is sufficient — run `python main.py` to start.", flush=True)

    print(f"\n{'─'*50}", flush=True)
    print(f"Done in {time.time()-t_start:.1f}s", flush=True)
    print(f"Model saved → {args.sklearn_output}", flush=True)
    print("Run `python main.py` to start QuietReach.", flush=True)


if __name__ == "__main__":
    main()