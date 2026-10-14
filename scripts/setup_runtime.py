"""Install external runtime assets inside this checkout and its venv."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    if Path(sys.prefix).resolve() != (ROOT / "venv").resolve():
        raise SystemExit("Run this script with venv/Scripts/python.exe")
    scripts = Path(sys.executable).parent
    if not (scripts / "avcodec-61.dll").exists():
        url = ("https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/"
               "ffmpeg-7.1.1-full_build-shared.zip")
        archive = ROOT / "venv" / "ffmpeg-shared.zip"
        print("Downloading FFmpeg 7.1.1 shared runtime...", flush=True)
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with archive.open("wb") as target:
                for block in response.iter_content(1024 * 1024):
                    target.write(block)
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                path = Path(entry.filename)
                if path.parent.name == "bin" and path.suffix in {".exe", ".dll"}:
                    with bundle.open(entry) as source, (scripts / path.name).open("wb") as target:
                        shutil.copyfileobj(source, target)
        archive.unlink()
    vendor = ROOT / "vendor" / "LaughterSegmentation"
    if not (vendor / "train" / "model.py").exists():
        subprocess.run(["git", "clone", "https://github.com/omine-me/LaughterSegmentation.git",
                        str(vendor)], check=True)
        subprocess.run(['git', '-C', str(vendor), 'checkout',
                        'a525292d26f744e14624e3a2f1fb5e3c7858d7b3'], check=True)
    print("Downloading laughter segmentation checkpoint...", flush=True)
    checkpoint = hf_hub_download("omine-me/LaughterSegmentation", "model.safetensors",
                                 local_dir=ROOT / "laughter" / "models")
    print(f"Checkpoint: {checkpoint}", flush=True)
    subprocess.run([str(scripts / "ffmpeg.exe"), "-version"], check=True)


if __name__ == "__main__":
    main()
