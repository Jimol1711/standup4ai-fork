"""Resumable Windows/venv implementation of README steps 2 through 6.

The IS corpus uses the public laughter checkpoint. ASR candidates are written
separately; the unpublished RF model/manual annotations are never fabricated.
"""
import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

from runtime import DATA_ROOT, ROOT, local_model
import pandas as pd

STAGES = ['download', 'audio', 'whisper', 'whisperx', 'laughter', 'labels']


def video_id(url):
    parsed = urlparse(url)
    value = parse_qs(parsed.query).get('v', [parsed.path.rsplit('/', 1)[-1]])[0]
    if len(value) != 11 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise ValueError(f'Invalid YouTube video URL: {url}')
    return value


def read_manifest(args):
    manifest = pd.read_csv(args.manifest).drop_duplicates('url').copy()
    for column in ['url', 'lang']:
        if column not in manifest:
            raise ValueError(f'Manifest missing {column}')
    manifest['region'] = manifest.get('region', manifest['lang']).fillna(manifest['lang'])
    if args.region:
        manifest = manifest[manifest.region.isin(args.region)]
    if args.limit:
        manifest = manifest.head(args.limit)
    if manifest.empty:
        raise ValueError('No videos selected')
    manifest['video_id'] = manifest.url.map(video_id)
    return manifest


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def record(args, stage, row, status, error=None):
    entry = {'stage': stage, 'video_id': row.video_id, 'region': row.region,
             'status': status, 'error': str(error) if error else None}
    with (args.data_root / 'events.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(entry, ensure_ascii=False) + '\n')


def download(args, manifest):
    import yt_dlp
    successes = 0
    for row in manifest.itertuples():
        folder = args.data_root / 'videos' / row.region
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f'{row.video_id}.mp4'
        if target.exists():
            successes += 1
            continue
        if shutil.disk_usage(args.data_root).free < args.min_free_gb * 2**30:
            raise RuntimeError('Insufficient disk space; free space then resume this command')
        last_progress = [0.]
        def progress(info):
            now = time.monotonic()
            if info['status'] == 'finished' or now - last_progress[0] >= 10:
                last_progress[0] = now
                total = info.get('total_bytes') or info.get('total_bytes_estimate')
                amount = info.get('downloaded_bytes', 0)
                percent = f'{100 * amount / total:.1f}%' if total else f'{amount / 2**20:.1f} MiB'
                print(f'[download {row.video_id}] {info["status"]}: {percent}', flush=True)
        opts = {
            'format': f'bv*[height<={args.max_height}][ext=mp4]+ba[ext=m4a]/b[height<={args.max_height}][ext=mp4]/b[height<={args.max_height}]',
            'outtmpl': str(folder / '%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'ffmpeg_location': str(Path(sys.executable).parent),
            'writesubtitles': not getattr(args, 'skip_subtitles', False),
            'writeautomaticsub': not getattr(args, 'skip_subtitles', False),
            'subtitleslangs': [row.lang], 'noplaylist': True,
            'retries': 3, 'fragment_retries': 3, 'socket_timeout': 30,
            'js_runtimes': {'node': {}}, 'noprogress': True,
            'progress_hooks': [progress],
        }
        if args.cookies:
            opts['cookiefile'] = str(args.cookies)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([row.url])
            # Downloads whose only available container is WebM need an MP4 remux.
            if not target.exists():
                candidates = [folder / f'{row.video_id}.{ext}' for ext in ['webm', 'mkv']]
                source = next((p for p in candidates if p.exists()), None)
                if source:
                    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-c', 'copy', str(target)], check=True)
            if not target.exists():
                raise RuntimeError('Downloader did not create a video file')
            successes += 1
            record(args, 'download', row, 'complete')
        except Exception as error:
            logging.exception('Download failed: %s', row.video_id)
            record(args, 'download', row, 'failed', error)
    if not successes:
        raise RuntimeError('No videos downloaded. See events.jsonl for availability/authentication errors.')


def audio(args, manifest):
    for row in manifest.itertuples():
        source = args.data_root / 'videos' / row.region / f'{row.video_id}.mp4'
        target = args.data_root / 'audio' / row.region / f'{row.video_id}.wav'
        if not source.exists() or target.exists():
            continue
        if shutil.disk_usage(args.data_root).free < args.min_free_gb * 2**30:
            raise RuntimeError('Insufficient disk space for audio; free space then resume')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp.wav')
        try:
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-vn',
                            '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', str(temporary)], check=True)
            temporary.replace(target)
            record(args, 'audio', row, 'complete')
        except Exception as error:
            record(args, 'audio', row, 'failed', error)


def whisper(args, manifest):
    from src.asr_pipeline import setup_model, process_batch
    pending = [r for r in manifest.itertuples()
               if (args.data_root / 'audio' / r.region / f'{r.video_id}.wav').exists()
               and not (args.data_root / 'transcript' / r.region / 'whisper' / f'{r.video_id}.json').exists()]
    if not pending:
        return
    pipe = setup_model(local_model(args.whisper_model), args.device)
    for row in pending:
        try:
            process_batch(pipe, [str(args.data_root / 'audio' / row.region / f'{row.video_id}.wav')],
                          str(args.data_root / 'transcript' / row.region / 'whisper'),
                          row.lang, 1, args.whisper_model, 'word')
            record(args, 'whisper', row, 'complete')
        except Exception as error:
            logging.exception('Whisper failed: %s', row.video_id)
            record(args, 'whisper', row, 'failed', error)


def whisperx_stage(args, manifest):
    from ASR.whisperX import whisperX_main
    for region, group in manifest.groupby('region', sort=False):
        source = args.data_root / 'audio' / region
        target = args.data_root / 'transcript' / region / 'whisperx'
        pending = [r for r in group.itertuples() if (source / f'{r.video_id}.wav').exists()
                   and not (target / f'{r.video_id}.csv').exists()]
        if not pending:
            continue
        by_id = {r.video_id: r for r in pending}
        whisperX_main([r.video_id for r in pending], str(source) + os.sep, str(target) + os.sep,
                      model_name=args.whisperx_model, lang=group.iloc[0]['lang'],
                      device=args.device.replace(':0', ''), batch_size=1,
                      compute_type='int8' if args.device == 'cpu' else 'float32',
                      on_error=lambda identifier, error: record(args, 'whisperx', by_id[identifier], 'failed', error))
        for row in pending:
            if (target / f'{row.video_id}.csv').exists():
                record(args, 'whisperx', row, 'complete')


def laughter(args, manifest):
    from laughter.inference_val import main
    for region, group in manifest.groupby('region', sort=False):
        source = args.data_root / 'audio' / region
        target = args.data_root / 'laughter_detection' / region
        pending = [r for r in group.itertuples() if (source / f'{r.video_id}.wav').exists()
                   and not (target / f'{r.video_id}.json').exists()]
        if not pending:
            continue
        main(str(source), str(target), str(args.laughter_model), batch_size=1, device=args.device)
        for row in pending:
            record(args, 'laughter', row, 'complete')


def labels(args, manifest):
    from ASR.create_validation_videos import process_file
    from labeling.create_seq_labels import df2json, label_words_with_laughter
    for row in manifest.itertuples():
        transcript = args.data_root / 'transcript' / row.region
        detection = args.data_root / 'laughter_detection' / row.region
        inputs = [transcript / 'whisperx' / f'{row.video_id}.csv',
                  transcript / 'whisper' / f'{row.video_id}.json', detection / f'{row.video_id}.json']
        if not all(p.exists() for p in inputs):
            continue
        mixed = transcript / 'mixed' / f'{row.video_id}.csv'
        updated = detection / 'Updated' / f'{row.video_id}.csv'
        target = args.data_root / 'dataset' / row.region / 'IS' / 'all' / f'{row.video_id}.csv'
        if target.exists():
            continue
        for path in [mixed, updated, target]:
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            process_file(*inputs[:2], mixed, inputs[2], updated, t_min=0.3)
            words = pd.read_csv(mixed)[['text', 't0', 't1', 't']]
            with inputs[2].open(encoding='utf-8') as source:
                events = json.load(source)
            tagged = label_words_with_laughter(df2json(words), events, verbose=False)
            temporary = target.with_suffix('.tmp.csv')
            pd.DataFrame(tagged, columns=['text', 'timestamp', 'label']).to_csv(temporary, index=False)
            temporary.replace(target)
            record(args, 'labels', row, 'complete')
        except Exception as error:
            logging.exception('Label generation failed: %s', row.video_id)
            record(args, 'labels', row, 'failed', error)


def status(args, manifest):
    counts = Counter()
    for row in manifest.itertuples():
        paths = {'download': f'videos/{row.region}/{row.video_id}.mp4',
                 'audio': f'audio/{row.region}/{row.video_id}.wav',
                 'whisper': f'transcript/{row.region}/whisper/{row.video_id}.json',
                 'whisperx': f'transcript/{row.region}/whisperx/{row.video_id}.csv',
                 'laughter': f'laughter_detection/{row.region}/{row.video_id}.json',
                 'labels': f'dataset/{row.region}/IS/all/{row.video_id}.csv'}
        for stage, path in paths.items():
            counts[stage] += (args.data_root / path).is_file()
    report = {'selected': len(manifest), 'completed': dict(counts),
              'corpus': 'IS (public laughter model)',
              'enhanced_filtered': 'Requires unpublished RF model and author annotations'}
    save_json(args.data_root / 'status.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'CSV_clean/StandUp4AI_v1.csv')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--region', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--stage', choices=STAGES + ['all', 'status'], default='all')
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--whisper-model', default='openai/whisper-large-v3')
    parser.add_argument('--whisperx-model', default='large-v3')
    parser.add_argument('--laughter-model', type=Path, default=ROOT / 'laughter/models/model.safetensors')
    parser.add_argument('--cookies', type=Path, help='Optional Netscape cookies file; never committed')
    parser.add_argument('--skip-subtitles', action='store_true', help='Do not fetch subtitles for audio-only labeling')
    parser.add_argument('--max-height', type=int, default=720)
    parser.add_argument('--min-free-gb', type=float, default=15)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / 'venv').resolve():
        parser.error('Use venv/Scripts/python.exe')
    args.data_root = args.data_root.resolve()
    args.data_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    manifest = read_manifest(args)
    if args.stage == 'status':
        status(args, manifest)
        return
    if args.stage == 'all':
        # Separate processes release model RAM/VRAM before the next stage.
        for stage in STAGES:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]]
            if '--stage' in command:
                command[command.index('--stage') + 1] = stage
            else:
                command += ['--stage', stage]
            logging.info('Starting stage: %s', stage)
            subprocess.run(command, check=True)
        counts = status(args, manifest)
        if counts['labels'] != len(manifest):
            raise SystemExit('Dataset partially generated; see events.jsonl and rerun to retry missing items.')
    else:
        functions = {'download': download, 'audio': audio, 'whisper': whisper,
                     'whisperx': whisperx_stage, 'laughter': laughter, 'labels': labels}
        try:
            functions[args.stage](args, manifest)
        finally:
            status(args, manifest)


if __name__ == '__main__':
    main()
