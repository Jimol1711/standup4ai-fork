# Windows setup and dataset generation

## Audio segment classification from downloaded videos

To expand the segment dataset to all URLs, preserving completed clips:

```powershell
.\scripts\start_full_audio_dataset.ps1
Get-Content data\audio_segments\download.log -Tail 20 -Wait
```

The full job downloads videos and labels completed downloads in batches. It does not run Whisper transcription. Press Ctrl+C to stop viewing the log; the background job continues. For counts, run `Get-Content data\audio_segments\full_progress.json`. Labeling details are in `data/audio_segments/labeling.log`; orchestration messages are in `data/audio_segments/full_pipeline.log`. To stop the job and all child processes, run `taskkill /PID (Get-Content data\audio_segments\full_pipeline.pid) /T /F`.

Complete video recordings are stored under `data/standup/videos/<region>/`, extracted WAV recordings under `data/standup/audio/<region>/`, and labeled clips under `data/audio_segments/clips/<region>/<video_id>/`. The dataset CSV remains `data/audio_segments/segments.csv`. Download errors are recorded in `data/standup/events.jsonl`; deleted/private/restricted videos may prevent reaching the full target. Rerun the launcher after a finished or interrupted run to retry missing downloads. Keep the computer awake and online while the job runs.

`build_audio_dataset.py` uses only completed local videos. It extracts their audio, runs the public laughter checkpoint, and writes `data/audio_segments/segments.csv` plus actual WAV clips under `data/audio_segments/clips`. No video downloads or ASR stages run. Labels are `laughter` (1) and `not_laughter` (0), with an explicit any-overlap rule on consecutive 5-second segments. The last segment of each video may be shorter.

```powershell
# Resume the selected videos; do not run a second copy while the job is active:
.\venv\Scripts\python.exe -u build_audio_dataset.py --device cuda:0
Get-Content data\audio_segments\status.json
Get-Content data\audio_segments\pipeline.stderr.log -Tail 20
```

`selected_videos.csv` freezes the 81 videos available at the start of this job. Completed per-video CSVs and clips are retained on resume. `segments.csv` is refreshed after each completed video, so a partial dataset is usable while processing continues. Keep clips from each source video in a single train/validation/test partition. Labels indicate predicted laughter, not a human judgment that the audio is comedic.

The generated CSV datasets and their metadata are tracked in Git. Downloaded media, WAV clips, model weights, virtual environments, and job logs remain local. The CSV `audio_path` values refer to clips under `data/audio_segments`; a fresh clone needs those clips regenerated or copied separately.

Use Python 3.11 and the repository's `venv`. All commands below run from the repository root in PowerShell.

```powershell
# Only if venv does not exist:
uv python install 3.11
uv venv --python 3.11 --seed venv

.\venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe scripts\setup_runtime.py
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\Activate.ps1
```

The runtime setup downloads FFmpeg 7.1.1 executables and shared DLLs into `venv/Scripts`, clones the public [LaughterSegmentation repository](https://github.com/omine-me/LaughterSegmentation), and downloads its 1.26 GB checkpoint into `laughter/models`. Its pretrained checkpoint is available for research use. Node.js 20+ must be on PATH for yt-dlp's JavaScript challenge solver; Node 24 is installed on this machine. Git must also be on PATH.

## Full run

The supplied `CSV_clean/StandUp4AI_v1.csv` already implements URL collection for the original dataset: 3,617 unique URLs, seven languages, nine regional groups, 334.24 hours. There is no need to scrape the channels again unless expanding the dataset.

```powershell
# Foreground; resumes files already completed:
.\venv\Scripts\python.exe -u run_dataset.py

# Or start a hidden background process that continues when this chat ends:
.\scripts\start_dataset.ps1

# Current counts and pipeline logs:
.\venv\Scripts\python.exe run_dataset.py --stage status
Get-Content data\standup\pipeline.stderr.log -Tail 30
Get-Content data\standup\pipeline.stdout.log -Tail 30
```

Do not start a second pipeline against the same data directory. The background launcher checks its PID file. To stop the background process and its child stage:

```powershell
taskkill /PID (Get-Content data\standup\pipeline.pid) /T /F
```

Keep the machine awake and connected. CPU inference with the full models is slow. CPU is the default because the installed GTX 1050 Ti has only 4 GB VRAM; CUDA with float32 is an explicit `--device cuda:0` option, requiring enough memory for the chosen models. The CUDA PyTorch build itself was verified on this GPU.

The pipeline downloads MP4 videos and available subtitles at up to 720p, extracts mono 16 kHz WAV audio, runs `openai/whisper-large-v3`, runs WhisperX `large-v3` with Silero VAD and word alignment, runs the public laughter model, fixes timestamps using both transcripts, writes ASR candidate intervals, and creates the base automatic `IS` corpus. Use `--max-height` to change video resolution. A 15 GB free-space guard stops downloads before filling the drive; the full corpus needs substantial disk space.

```text
data/standup/
  videos/<region>/<youtube-id>.mp4
  audio/<region>/<youtube-id>.wav
  transcript/<region>/whisper/<youtube-id>.json
  transcript/<region>/whisperx/<youtube-id>.csv
  transcript/<region>/mixed/<youtube-id>.csv
  laughter_detection/<region>/<youtube-id>.json
  laughter_detection/<region>/Updated/<youtube-id>.csv
  dataset/<region>/IS/all/<youtube-id>.csv
  events.jsonl
  status.json
```

Each stage runs in a separate process to release model memory before the next stage. Outputs are skipped on rerun. `events.jsonl` records per-video failures, including deleted/private/region-blocked videos. Such videos cannot be recovered by installing dependencies. The pipeline reports partial completion when some URLs could not produce labels. If YouTube requires authentication, supply your own Netscape-format cookies file with `--cookies <path>`; credentials are never read automatically.

Run individual stages or a subset:

```powershell
.\venv\Scripts\python.exe run_dataset.py --stage audio
.\venv\Scripts\python.exe run_dataset.py --stage whisperx --region es es_latam
.\venv\Scripts\python.exe run_dataset.py --region en_us --limit 1 --data-root data\trial
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

`data/smoke` is an isolated verification directory containing an eight-second clip; it is not part of the generated corpus.

Verified on 2026-10-14: a real video downloaded with subtitles, FFmpeg extracted mono 16 kHz audio, the full Whisper large-v3 and WhisperX large-v3 models transcribed/aligned an eight-second clip, the public laughter checkpoint produced an interval, and label generation produced a valid 25-word CSV. Dependency checks and the three pipeline regression tests passed. The full 3,617-URL job was then left running; this smoke check does not mean the full corpus is complete.

## Published enhanced corpus and training

The checkout does not include the random-forest candidate validator from README step 5.3, its training annotations, the 70 manually annotated test videos, or the released train/dev/test partitions. Consequently, setup cannot reproduce the paper's `Filtered` or `Manual` datasets or its exact reported scores. `Updated` CSVs explicitly distinguish `Initial` and `Added` candidates; added candidates are not treated as validated positives. Request those assets from the authors using the contact in README.md. No messages are sent automatically.

The training dependencies are installed, and the training script uses the local data root and current Transformers API. Training still requires real train/dev/test partitions; the automatic pipeline writes `IS/all` and does not invent the paper's splits. The optional `STANDUP_DATA_ROOT` environment variable changes data paths for the label-generation and training utilities; `run_dataset.py` also accepts `--data-root`. Model inference files are stored directly under `data/models` to avoid Windows symlink privileges.
