"""Download CSV URLs without changing the manifest or guessing filenames."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from runtime import DATA_ROOT
from run_dataset import download, video_id


def download_videos(sub_lang, urls_file, skip_alread_download=True, data_root=DATA_ROOT):
    path = Path(urls_file)
    if path.suffix.lower() == '.csv':
        manifest = pd.read_csv(path)
    else:
        urls = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip() and line.strip() != 'url']
        manifest = pd.DataFrame({'url': urls})
    manifest['lang'] = sub_lang.split('_')[0]
    manifest['region'] = sub_lang
    manifest['video_id'] = manifest.url.map(video_id)
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(data_root=root, min_free_gb=15, max_height=720, cookies=None)
    download(args, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subtitle_language')
    parser.add_argument('urls_file', type=Path)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    download_videos(args.subtitle_language, args.urls_file, data_root=args.data_root)
