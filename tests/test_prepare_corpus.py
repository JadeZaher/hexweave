"""Corpus preparation: complete stories only, every Nth held out, END-terminated, manifest pins the bytes."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_corpus


class PrepareCorpusTest(unittest.TestCase):
    def test_split_drops_fragment_holds_out_every_nth_and_pins_files(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        source = root / "source.txt"
        source.write_bytes(b"tail of a cut story\n<|endoftext|>\nStory one.\n<|endoftext|>\n"
                           b"Story two.\n<|endoftext|>\nStory three.\n")
        manifest = prepare_corpus.prepare(source, root / "corpus", every=2)
        self.assertEqual((root / "corpus/heldout.txt").read_bytes(), b"Story one.\x03Story three.\x03")
        self.assertEqual((root / "corpus/train.txt").read_bytes(), b"Story two.\x03")
        for name, entry in manifest["files"].items():
            data = (root / "corpus" / name).read_bytes()
            self.assertEqual(entry, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.assertEqual(json.loads((root / "corpus/manifest.json").read_text(encoding="utf-8")), manifest)


if __name__ == "__main__":
    unittest.main()
