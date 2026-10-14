import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

import pandas as pd
from build_audio_dataset import export_video, merge_intervals


class AudioSegmentTests(unittest.TestCase):
    def test_overlapping_events_are_counted_once_and_clipped(self):
        events = {'a': {'start_sec': -1, 'end_sec': 2},
                  'b': {'start_sec': 1, 'end_sec': 5}}
        self.assertEqual(merge_intervals(events, 3), [[0., 3]])

    def test_exports_real_clips_with_positive_negative_and_short_tail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'example.wav'
            with wave.open(str(source), 'wb') as audio:
                audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\0\0' * (12 * 16000))
            events = root / 'events.json'
            events.write_text(json.dumps({'0': {'start_sec': 5., 'end_sec': 6.}}))
            export_video(source, events, root, 'en_us', 5.)
            frame = pd.read_csv(root / 'per_video/en_us/example.csv')
            self.assertEqual(frame.label.tolist(), ['not_laughter', 'laughter', 'not_laughter'])
            self.assertEqual(frame.duration_sec.tolist(), [5., 5., 2.])
            self.assertEqual(frame.laughter_overlap_sec.tolist(), [0., 1., 0.])
            for row in frame.itertuples():
                with wave.open(str(root / row.audio_path), 'rb') as clip:
                    self.assertAlmostEqual(clip.getnframes() / clip.getframerate(), row.duration_sec)


if __name__ == '__main__':
    unittest.main()
