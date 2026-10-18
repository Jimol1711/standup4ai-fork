"""Resume all video downloads and expand the binary audio-segment dataset."""
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from runtime import ROOT, DATA_ROOT
from run_dataset import save_json

OUTPUT = ROOT / 'data/audio_segments'


def snapshot(manifest, downloader, labeler):
    downloaded = {row.video_id for row in manifest.itertuples()
                  if (DATA_ROOT / 'videos' / row.region / f'{row.video_id}.mp4').exists()}
    labeled = {row.video_id for row in manifest.itertuples()
               if (OUTPUT / 'per_video' / row.region / f'{row.video_id}.csv').exists()}
    failed = set()
    events = DATA_ROOT / 'events.jsonl'
    if events.exists():
        for line in events.read_text(encoding='utf-8').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('stage') == 'download' and event.get('status') == 'failed':
                failed.add(event['video_id'])
    progress = {'total_videos': len(manifest), 'downloaded_videos': len(downloaded),
                'remaining_downloads': len(manifest) - len(downloaded),
                'labeled_videos': len(labeled), 'failed_downloads': len(failed - downloaded),
                'download_running': downloader.poll() is None,
                'labeling_running': labeler is not None and labeler.poll() is None,
                'complete': len(labeled) == len(manifest)}
    save_json(OUTPUT / 'full_progress.json', progress)
    return progress


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    manifest = pd.read_csv(ROOT / 'CSV_clean/StandUp4AI_v1.csv').drop_duplicates('url')
    manifest['video_id'] = manifest.url.str.extract(r'[?&]v=([^&]+)')
    # Both log streams go to dedicated files so terminal tailing stays simple.
    with (OUTPUT / 'download.log').open('a', encoding='utf-8') as downloads, \
         (OUTPUT / 'labeling.log').open('a', encoding='utf-8') as labels:
        downloader = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run_dataset.py'),
                                       '--stage', 'download', '--skip-subtitles'],
                                      cwd=ROOT, stdout=downloads, stderr=subprocess.STDOUT)
        labeler = None
        processed_available = -1
        had_label_failure = False
        while True:
            progress = snapshot(manifest, downloader, labeler)
            logging.info('Downloaded %s/%s; labeled %s; pending/failed downloads %s/%s',
                         progress['downloaded_videos'], len(manifest), progress['labeled_videos'],
                         progress['remaining_downloads'], progress['failed_downloads'])
            if labeler is not None and labeler.poll() is not None:
                if labeler.returncode:
                    logging.error('Labeling batch failed; inspect labeling.log. Downloads continue.')
                    had_label_failure = True
                labeler = None
            if labeler is None and progress['downloaded_videos'] > processed_available:
                processed_available = progress['downloaded_videos']
                labeler = subprocess.Popen([sys.executable, '-u', str(ROOT / 'build_audio_dataset.py'),
                                            '--refresh-selection', '--expected-total', str(len(manifest)),
                                            '--device', 'cuda:0'], cwd=ROOT,
                                           stdout=labels, stderr=subprocess.STDOUT)
            if downloader.poll() is not None and labeler is None:
                final = snapshot(manifest, downloader, labeler)
                final['run_finished'] = True
                final['download_exit_code'] = downloader.returncode
                final['labeling_batch_failed'] = had_label_failure
                save_json(OUTPUT / 'full_progress.json', final)
                logging.info('Run finished: %s', json.dumps(final))
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
