# Automatically labeled audio segments

segments.csv has one row per consecutive 5-second WAV clip (final clips can be shorter). audio_path is relative to this directory. label_id is 1 for laughter and 0 for not_laughter. Any overlap with a detected laughter interval makes a clip positive; overlap seconds and fraction are included. These are model predictions, not human annotations or judgments of humor. No ASR is needed for this audio classification dataset.

selected_videos.csv freezes the already-downloaded source videos. status.json tracks progress; segments.csv is refreshed after each completed video. Clips from the same source video must stay in the same train/validation/test partition to avoid leakage.
