from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import finish_dataset


class FullDatasetProgressTests(unittest.TestCase):
    def test_counts_finished_files_and_excludes_recovered_download_failures(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / 'videos/en_us'
            labels = root / 'output/per_video/en_us'
            videos.mkdir(parents=True)
            labels.mkdir(parents=True)
            (videos / 'first.mp4').touch()
            (videos / 'second.mp4.part').touch()
            (labels / 'first.csv').touch()
            (root / 'events.jsonl').write_text(
                '{"stage":"download","status":"failed","video_id":"first"}\n'
                '{"stage":"download","status":"failed","video_id":"second"}\n'
                '{"unfinished":', encoding='utf-8')
            manifest = pd.DataFrame({'video_id': ['first', 'second'], 'region': ['en_us'] * 2})
            with patch.object(finish_dataset, 'DATA_ROOT', root), \
                 patch.object(finish_dataset, 'OUTPUT', root / 'output'):
                result = finish_dataset.snapshot(manifest, SimpleNamespace(poll=lambda: None), None)
            self.assertEqual(result['downloaded_videos'], 1)
            self.assertEqual(result['labeled_videos'], 1)
            self.assertEqual(result['failed_downloads'], 1)
            self.assertFalse(result['complete'])


if __name__ == '__main__':
    unittest.main()
