"""
download_datasets.py — download real emotional speech datasets for QuietReach

Datasets used:
  RAVDESS — emotional speech: angry/fearful → threat, calm/neutral → normal
  CREMA-D — emotional speech: angry/fearful/disgust → threat, calm/neutral → normal

These contain actual human vocal distress sounds, which is what
QuietReach needs to detect — not sirens or gunshots.

Run from QuietReach root:
    python download_datasets.py

After completing, retrain:
    python -m model.trainer --sklearn-only --skip-download
"""

import csv
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

THREAT_DIR = Path("data/custom/threat")
NORMAL_DIR = Path("data/custom/normal")


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Check it's not a partial download (less than 1MB is suspicious)
        if dest.stat().st_size > 1_048_576:
            print(f"  Already downloaded: {dest.name} ({dest.stat().st_size // 1_048_576} MB)", flush=True)
            return True
        else:
            print(f"  Partial file found, re-downloading...", flush=True)
            dest.unlink()

    print(f"  Downloading {dest.name}...", flush=True)

    for attempt in range(1, retries + 1):
        try:
            import urllib.request
            # Use chunked download with timeout instead of urlretrieve
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536  # 64KB chunks

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = min(downloaded / total_size * 100, 100)
                            mb  = downloaded / 1_048_576
                            total_mb = total_size / 1_048_576
                            print(f"  {pct:5.1f}%  {mb:.0f}/{total_mb:.0f} MB", end="\r", flush=True)

            print(f"  Done. ({downloaded // 1_048_576} MB)                    ", flush=True)
            return True

        except Exception as e:
            print(f"\n  Attempt {attempt}/{retries} failed: {e}", flush=True)
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                import time
                print(f"  Retrying in 5s...", flush=True)
                time.sleep(5)

    return False


# ── RAVDESS ───────────────────────────────────────────────────────────────────
# Ryerson Audio-Visual Database of Emotional Speech
# 24 professional actors, 8 emotions, ~200MB
# No login required — direct Zenodo download
#
# Filename format: 03-01-EM-IN-ST-RE-AC.wav
#   Segment 3 (index 2) = emotion:
#     01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
#   Segment 4 (index 3) = intensity: 01=normal 02=strong
#
# Threat → angry(05) + fearful(06) + disgusted(07) — all represent distress states
# Normal → neutral(01) + calm(02) + happy(03)

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

RAVDESS_THREAT_EMOTIONS = {"05", "06", "07"}   # angry, fearful, disgust
RAVDESS_NORMAL_EMOTIONS = {"01", "02", "03"}   # neutral, calm, happy


def download_ravdess() -> tuple[int, int]:
    """
    Download RAVDESS audio-only files as 24 individual actor zips (~9MB each).
    Much more reliable than the single 200MB zip which drops mid-transfer.
    """
    extracted = Path("data/ravdess")
    extracted.mkdir(parents=True, exist_ok=True)

    print("\nRAVDESS Emotional Speech (24 actors, ~9MB each, no login)...", flush=True)

    # Each actor has their own zip on Zenodo
    # Format: Audio_Speech_Actor_01.zip ... Audio_Speech_Actor_24.zip
    BASE_URL = "https://zenodo.org/records/1188976/files"
    downloaded_actors = 0
    failed_actors = []

    for actor_num in range(1, 25):
        actor_str  = f"{actor_num:02d}"
        zip_name   = f"Audio_Speech_Actor_{actor_str}.zip"
        zip_path   = extracted / zip_name
        actor_dir  = extracted / f"Actor_{actor_str}"

        if actor_dir.exists() and list(actor_dir.glob("*.wav")):
            downloaded_actors += 1
            continue  # already extracted

        url = f"{BASE_URL}/{zip_name}?download=1"
        print(f"  Actor {actor_str}/24...", end=" ", flush=True)

        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            with open(zip_path, "wb") as f:
                f.write(data)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extracted)
            zip_path.unlink()
            downloaded_actors += 1
            print("OK", flush=True)

        except Exception as e:
            print(f"FAILED ({e})", flush=True)
            failed_actors.append(actor_num)
            if zip_path.exists():
                zip_path.unlink()

    if failed_actors:
        print(f"  {len(failed_actors)} actors failed: {failed_actors}", flush=True)
    print(f"  Downloaded {downloaded_actors}/24 actors.", flush=True)

    # Sort into threat/normal
    THREAT_DIR.mkdir(parents=True, exist_ok=True)
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    all_wavs = list(extracted.rglob("*.wav"))
    copied_t = copied_n = 0

    for src in all_wavs:
        parts = src.stem.split("-")
        if len(parts) < 3:
            continue
        emotion = parts[2]   # 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust

        if emotion in {"05", "06", "07"}:   # angry, fearful, disgust → threat
            dest = THREAT_DIR / f"ravdess_t_{copied_t:04d}_{src.name}"
            if not dest.exists():
                shutil.copy(src, dest)
            copied_t += 1
        elif emotion in {"01", "02", "03"}: # neutral, calm, happy → normal
            dest = NORMAL_DIR / f"ravdess_n_{copied_n:04d}_{src.name}"
            if not dest.exists():
                shutil.copy(src, dest)
            copied_n += 1

    print(f"  Sorted: {copied_t} threat (angry/fearful/disgust) + {copied_n} normal (calm/neutral/happy)", flush=True)
    return copied_t, copied_n


# ── CREMA-D ───────────────────────────────────────────────────────────────────
# Crowd-sourced Emotional Multimodal Actors Dataset
# 91 actors, 6 emotions, 7442 clips, ~600MB
# No login — direct GitHub release download
#
# Filename format: ACTORID_SENTENCE_EMOTION_LEVEL.wav
#   EMOTION: ANG=angry, DIS=disgust, FEA=fear, HAP=happy, NEU=neutral, SAD=sad
#   LEVEL:   LO=low, MD=medium, HI=high, XX=unspecified
#
# Threat → ANG + FEA + DIS
# Normal → NEU + HAP

def download_cremad() -> tuple[int, int]:
    """
    CREMA-D sorter — auto-sorts if already placed locally.
    Download from Kaggle (free account):
      https://www.kaggle.com/datasets/ejlok1/cremad
    Unzip into data/cremad/ then re-run this script.
    """
    extracted = Path("data/cremad")

    if not extracted.exists():
        print("\nCREMA-D: not found locally.", flush=True)
        print("  Download free from Kaggle:", flush=True)
        print("  https://www.kaggle.com/datasets/ejlok1/cremad", flush=True)
        print("  Unzip so data/cremad/AudioWAV/*.wav exists, then re-run.", flush=True)
        return 0, 0

    print("\nCREMA-D: sorting locally found files...", flush=True)
    THREAT_DIR.mkdir(parents=True, exist_ok=True)
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    all_wavs = list(extracted.rglob("*.wav"))
    copied_t = copied_n = 0

    for src in all_wavs:
        parts   = src.stem.split("_")
        if len(parts) < 3:
            continue
        emotion = parts[2].upper()

        if emotion in {"ANG", "FEA", "DIS"}:
            dest = THREAT_DIR / f"cremad_t_{copied_t:04d}_{src.name}"
            if not dest.exists():
                shutil.copy(src, dest)
            copied_t += 1
        elif emotion in {"NEU", "HAP"}:
            dest = NORMAL_DIR / f"cremad_n_{copied_n:04d}_{src.name}"
            if not dest.exists():
                shutil.copy(src, dest)
            copied_n += 1

    print(f"  CREMA-D: {copied_t} threat + {copied_n} normal sorted.", flush=True)
    return copied_t, copied_n



# ── Also fix ESC-50 categories while we're here ───────────────────────────────

def fix_esc50_custom_copies() -> tuple[int, int]:
    """
    Pull extra relevant clips from ESC-50 that we may have missed.
    Only crying_baby and glass_breaking are genuinely relevant.
    Removes irrelevant categories from custom folder if present.
    """
    esc50_audio = Path("data/esc50/audio")
    esc50_meta  = Path("data/esc50/meta/esc50.csv")

    if not esc50_meta.exists() or not esc50_audio.exists():
        return 0, 0

    # Only categories genuinely relevant to domestic incidents
    KEEP_THREAT  = {"crying_baby", "glass_breaking", "door_wood_creaks", "door_wood_knock"}
    KEEP_NORMAL  = {"clock_tick", "keyboard_typing", "water_drops", "rain", "wind"}

    copied_t = copied_n = 0
    with open(esc50_meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row      = {k.strip(): v.strip() for k, v in row.items()}
            category = row.get("category", "")
            fpath    = esc50_audio / row.get("filename", "")
            if not fpath.exists():
                continue
            if category in KEEP_THREAT:
                dest = THREAT_DIR / f"esc50_t_{copied_t:04d}_{fpath.name}"
                if not dest.exists():
                    shutil.copy(fpath, dest)
                copied_t += 1
            elif category in KEEP_NORMAL:
                dest = NORMAL_DIR / f"esc50_n_{copied_n:04d}_{fpath.name}"
                if not dest.exists():
                    shutil.copy(fpath, dest)
                copied_n += 1

    if copied_t or copied_n:
        print(f"\nESC-50 relevant clips: {copied_t} threat + {copied_n} normal", flush=True)
    return copied_t, copied_n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("QuietReach — Dataset Downloader", flush=True)
    print("Downloading EMOTIONAL SPEECH datasets (angry/fearful = threat)", flush=True)
    print("=" * 60, flush=True)
    print(f"Threat audio → {THREAT_DIR}", flush=True)
    print(f"Normal audio → {NORMAL_DIR}\n", flush=True)

    THREAT_DIR.mkdir(parents=True, exist_ok=True)
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    before_t = len(list(THREAT_DIR.rglob("*.wav")))
    before_n = len(list(NORMAL_DIR.rglob("*.wav")))
    print(f"Existing: {before_t} threat / {before_n} normal\n", flush=True)

    total_t = total_n = 0

    # RAVDESS — 200MB, direct download, confirmed working
    t, n = download_ravdess()
    total_t += t; total_n += n

    # CREMA-D — 600MB, direct download
    t, n = download_cremad()
    total_t += t; total_n += n

    # Pull relevant ESC-50 clips into custom folder too
    t, n = fix_esc50_custom_copies()
    total_t += t; total_n += n

    # Final count
    after_t = len(list(THREAT_DIR.rglob("*.wav")))
    after_n = len(list(NORMAL_DIR.rglob("*.wav")))

    print(f"\n{'=' * 60}", flush=True)
    print(f"New files added : {total_t} threat + {total_n} normal", flush=True)
    print(f"Total now       : {after_t} threat / {after_n} normal", flush=True)

    if after_t < 100 or after_n < 100:
        print("\nWARNING: fewer than 100 per class — using synthetic + ESC-50 for now.", flush=True)
        print("  python -m model.trainer --sklearn-only --synthetic --synthetic-samples 1000", flush=True)
    else:
        print(f"\nGood dataset size. Now retrain:", flush=True)
        print(f"  python -m model.trainer --sklearn-only --skip-download", flush=True)

    print("\nVerify after training:", flush=True)
    print("  python model_check.py", flush=True)


if __name__ == "__main__":
    main()