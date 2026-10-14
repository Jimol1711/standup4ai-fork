import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd

from labeling.create_seq_labels import label_words_with_laughter
from run_dataset import labels, read_manifest


class DatasetPipelineTests(unittest.TestCase):
    def test_laughter_before_first_word_does_not_label_last_word(self):
        words = [{'text': 'first', 'timestamp': [1., 2.]},
                 {'text': 'last', 'timestamp': [3., 4.]}]
        result = label_words_with_laughter(words, {'0': {'start_sec': 0., 'end_sec': .5}}, False)
        self.assertEqual([r['label'] for r in result], ['U', 'O'])

    def test_labels_use_named_event_fields_and_require_all_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / 'transcript/en_us'
            detection = root / 'laughter_detection/en_us'
            for path in [transcript / 'whisperx', transcript / 'whisper', detection]:
                path.mkdir(parents=True)
            identifier = 'Yp9LaRbLyLo'
            words = pd.DataFrame({'text': ['one', 'two', 'three'], 'score': [.9] * 3,
                                  't0': [1., 2., 3.], 't1': [1.5, 2.5, 3.5], 't': [.5] * 3})
            words.to_csv(transcript / f'whisperx/{identifier}.csv')
            chunks = [{'text': text, 'timestamp': [start, end]}
                      for text, start, end in zip(words.text, words.t0, words.t1)]
            (transcript / f'whisper/{identifier}.json').write_text(json.dumps({'chunks': chunks}))
            manifest = pd.DataFrame({'video_id': [identifier], 'region': ['en_us'], 'lang': ['en']})
            target = root / f'dataset/en_us/IS/all/{identifier}.csv'
            labels(SimpleNamespace(data_root=root), manifest)
            self.assertFalse(target.exists())
            (detection / f'{identifier}.json').write_text(json.dumps({'0': {'end_sec': 2.8, 'start_sec': 1.2}}))
            labels(SimpleNamespace(data_root=root), manifest)
            self.assertEqual(pd.read_csv(target).label.tolist(), ['B', 'L', 'O'])
            self.assertEqual(pd.read_csv(detection / f'Updated/{identifier}.csv').t0.tolist(), [1.2])

    def test_manifest_deduplicates_and_keeps_regional_groups(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            pd.DataFrame({'url': ['https://www.youtube.com/watch?v=Yp9LaRbLyLo'] * 2,
                          'lang': ['en'] * 2, 'region': ['en_us'] * 2}).to_csv(path, index=False)
            rows = read_manifest(SimpleNamespace(manifest=path, region=None, limit=None))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.iloc[0].region, 'en_us')


if __name__ == '__main__':
    unittest.main()
