import gzip
import json
from pathlib import Path
import tempfile
import unittest

from anonymize_paths import anonymize_text, clean_pack, encoded, clean_json


class AnonymizePathsTests(unittest.TestCase):
    def test_paths_and_public_urls(self):
        for original, expected in [
            ('/local/home/alice/work', '/local/home/user/work'),
            ('/home/runner/.lake', '/home/user/.lake'),
            ('/Users/alice-smith/work', '/Users/user/work'),
            ('file:///home/alice/work', 'file:///home/user/work'),
            (r'C:\Users\Alice\work', r'C:\Users\user\work'),
        ]:
            self.assertEqual(anonymize_text(original), expected)
        for public in ['https://example.org/home/alice/work', 'relative/home/alice/work', '/home/user/work']:
            self.assertEqual(anonymize_text(public), public)
        self.assertEqual(clean_json(json.dumps({'path': r'c:\users\alice\work'}).encode()),
                         {'path': r'c:\users\user\work'})

    def test_trace_pack_keeps_steps_and_reindexes_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            pack = Path(folder) / 'agent.bin'
            chunks = [encoded({'0': {'o': '/home/a_long_personal_username/work'}}),
                      encoded({'1': {'o': 'unchanged output'}})]
            head = {'score': 42, 'steps': [{'p': 0}, {'p': 1}],
                    'pack': [[0, len(chunks[0])], [len(chunks[0]), len(chunks[1])]]}
            pack.write_bytes(b''.join(chunks))
            hp = pack.with_suffix('.json.gz'); hp.write_bytes(encoded(head))
            original = pack.read_bytes(), hp.read_bytes()
            self.assertTrue(clean_pack(pack, check=True))
            self.assertEqual((pack.read_bytes(), hp.read_bytes()), original)
            self.assertTrue(clean_pack(pack))
            updated = json.loads(gzip.decompress(hp.read_bytes()))
            self.assertEqual(updated['steps'], head['steps'])
            self.assertEqual(updated['score'], 42)
            content = pack.read_bytes()
            offset, length = updated['pack'][0]
            self.assertEqual(json.loads(gzip.decompress(content[offset:offset+length])),
                             {'0': {'o': '/home/user/work'}})
            offset, length = updated['pack'][1]
            self.assertEqual(content[offset:offset+length], chunks[1])
            self.assertFalse(clean_pack(pack, check=True))

    def test_source_pack_index(self):
        with tempfile.TemporaryDirectory() as folder:
            pack = Path(folder) / 'source.bin'
            chunk = encoded('-- built in /home/alice/project\n')
            pack.write_bytes(chunk)
            hp = pack.with_suffix('.json.gz'); hp.write_bytes(encoded({'Main.lean': [0, len(chunk)]}))
            self.assertTrue(clean_pack(pack))
            head = json.loads(gzip.decompress(hp.read_bytes()))
            self.assertEqual(head, {'Main.lean': [0, len(pack.read_bytes())]})
            self.assertEqual(json.loads(gzip.decompress(pack.read_bytes())), '-- built in /home/user/project\n')

    def test_invalid_pack_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as folder:
            pack = Path(folder) / 'agent.bin'
            chunk = encoded({'0': {'o': '/home/alice/work'}})
            pack.write_bytes(chunk)
            hp = pack.with_suffix('.json.gz'); hp.write_bytes(encoded({'pack': [[1, len(chunk)-1]]}))
            with self.assertRaises(ValueError):
                clean_pack(pack)
            self.assertEqual(pack.read_bytes(), chunk)


if __name__ == '__main__':
    unittest.main()
