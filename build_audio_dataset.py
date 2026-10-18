"""Label only already-downloaded videos and export fixed-duration WAV clips."""
import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import wave

from runtime import ROOT, DATA_ROOT
import pandas as pd
from run_dataset import save_json


def merge_intervals(events, duration):
    intervals = sorted((max(0., float(e['start_sec'])), min(duration, float(e['end_sec'])))
                       for e in events.values())
    merged = []
    for start, end in intervals:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def export_video(audio_path, event_path, output, region, segment_seconds):
    identifier = audio_path.stem
    target = output / 'per_video' / region / f'{identifier}.csv'
    if target.exists():
        return
    with event_path.open(encoding='utf-8') as source:
        events = json.load(source)
    clip_dir = output / 'clips' / region / identifier
    clip_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with wave.open(str(audio_path), 'rb') as audio:
        rate = audio.getframerate()
        total = audio.getnframes()
        frames_per_clip = round(segment_seconds * rate)
        duration = total / rate
        intervals = merge_intervals(events, duration)
        for index, first in enumerate(range(0, total, frames_per_clip)):
            count = min(frames_per_clip, total - first)
            start, end = first / rate, (first + count) / rate
            overlap = sum(max(0., min(end, b) - max(start, a)) for a, b in intervals)
            positive = overlap > 1e-9
            clip = clip_dir / f'{identifier}_{index:05d}.wav'
            temporary = clip.with_suffix('.tmp.wav')
            with wave.open(str(temporary), 'wb') as destination:
                destination.setparams(audio.getparams())
                destination.writeframes(audio.readframes(count))
            temporary.replace(clip)
            rows.append({'video_id': identifier, 'region': region, 'segment_index': index,
                         'audio_path': clip.relative_to(output).as_posix(),
                         'start_sec': start, 'end_sec': end, 'duration_sec': end - start,
                         'label': 'laughter' if positive else 'not_laughter',
                         'label_id': int(positive), 'laughter_overlap_sec': overlap,
                         'laughter_fraction': overlap / (end - start), 'label_source': 'automatic_omine'})
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp.csv')
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(target)


def combine(output, expected):
    files = sorted((output / 'per_video').glob('*/*.csv'))
    if files:
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        temporary = output / 'segments.tmp.csv'
        frame.to_csv(temporary, index=False)
        temporary.replace(output / 'segments.csv')
        report = {'selected_videos': expected, 'completed_videos': int(frame.video_id.nunique()),
                  'segments': len(frame), 'label_counts': frame.label.value_counts().to_dict(),
                  'complete': frame.video_id.nunique() == expected}
    else:
        report = {'selected_videos': expected, 'completed_videos': 0, 'segments': 0, 'complete': False}
    save_json(output / 'status.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/audio_segments')
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    parser.add_argument('--segment-seconds', type=float, default=5.)
    parser.add_argument('--refresh-selection', action='store_true', help='Include newly downloaded videos')
    parser.add_argument('--expected-total', type=int, help='Full target count while downloads continue')
    args = parser.parse_args()
    if args.segment_seconds <= 0:
        parser.error('--segment-seconds must be positive')
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous['segment_seconds'] != args.segment_seconds:
            parser.error('Use a different output directory when changing segment length')
    logging.basicConfig(level=logging.INFO)
    # Freeze the selection once. Never fetch a URL or include unfinished downloads.
    selection = output / 'selected_videos.csv'
    if selection.exists() and not args.refresh_selection:
        selected = pd.read_csv(selection)
    else:
        metadata = pd.read_csv(ROOT / 'CSV_clean/StandUp4AI_v1.csv')
        metadata['video_id'] = metadata.url.str.extract(r'[?&]v=([^&]+)')
        selected = metadata[[((args.data_root / 'videos' / row.region / f'{row.video_id}.mp4').exists())
                             for row in metadata.itertuples()]].copy()
        if selected.empty:
            raise RuntimeError('No completed downloaded videos found')
        selected.to_csv(selection, index=False)
    save_json(output / 'config.json', {'segment_seconds': args.segment_seconds,
              'positive_rule': 'any detected laughter overlap > 0 seconds',
              'checkpoint': 'omine-me/LaughterSegmentation', 'device': args.device,
              'downloads_enabled': False})
    (output / 'README.md').write_text(
        '# Automatically labeled audio segments\n\n'
        'segments.csv has one row per consecutive 5-second WAV clip (final clips can be shorter). '
        'audio_path is relative to this directory. label_id is 1 for laughter and 0 for not_laughter. '
        'Any overlap with a detected laughter interval makes a clip positive; overlap seconds and '
        'fraction are included. These are model predictions, not human annotations or judgments of humor. '
        'No ASR is needed for this audio classification dataset.\n\n'
        'selected_videos.csv freezes the already-downloaded source videos. status.json tracks progress; '
        'segments.csv is refreshed after each completed video. Clips from the same source video '
        'must stay in the same train/validation/test partition to avoid leakage.\n', encoding='utf-8')
    expected = args.expected_total or len(selected)
    combine(output, expected)
    for row in selected.itertuples():
        audio = args.data_root / 'audio' / row.region / f'{row.video_id}.wav'
        if audio.exists():
            continue
        if shutil.disk_usage(output).free < 15 * 2**30:
            raise RuntimeError('Less than 15 GB free disk space')
        audio.parent.mkdir(parents=True, exist_ok=True)
        temporary = audio.with_suffix('.tmp.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i',
                        str(args.data_root / 'videos' / row.region / f'{row.video_id}.mp4'),
                        '-vn', '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', str(temporary)], check=True)
        temporary.replace(audio)
    from laughter.inference_val import main as infer
    for region, group in selected.groupby('region'):
        if all((output / 'per_video' / region / f'{identifier}.csv').exists() for identifier in group.video_id):
            continue
        # Audio directory contains only extracted local videos; no download stage exists.
        def complete(audio_path, event_path):
            if audio_path.stem in set(group.video_id):
                export_video(audio_path, event_path, output, region, args.segment_seconds)
                combine(output, expected)
        infer(str(args.data_root / 'audio' / region),
              str(args.data_root / 'laughter_detection' / region),
              str(ROOT / 'laughter/models/model.safetensors'), batch_size=1,
              device=args.device, on_complete=complete)
    combine(output, expected)


if __name__ == '__main__':
    main()
