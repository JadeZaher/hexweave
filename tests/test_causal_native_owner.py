"""One Windows-only nontraining Job/STOP lifecycle spot check."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts import causal_resources


@unittest.skipUnless(os.name == "nt", "native Windows Job lifecycle")
class CausalNativeOwnerTests(unittest.TestCase):
    def test_one_suspended_child_job_stop_and_exact_reap(self):
        root = Path(__file__).resolve().parents[1]
        work = root / "work"
        work.mkdir(exist_ok=True)
        state = causal_resources.memory_status()
        needed = causal_resources.required_cushion(state["total_physical_bytes"]) + causal_resources.CHILD_LIMIT
        self.assertGreaterEqual(state["available_physical_bytes"], needed)
        with tempfile.TemporaryDirectory(prefix="causal-native-fixture-", dir=work) as name:
            directory = Path(name)
            ready, stop = directory / "ready", directory / "STOP"
            program = ("import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
                       "(p/'ready').write_text('ready'); end=time.monotonic()+4\n"
                       "while not (p/'STOP').exists() and time.monotonic()<end: time.sleep(.02)\n"
                       "sys.exit(0 if (p/'STOP').exists() else 2)")
            process, job = None, causal_resources.Job()
            try:
                process = subprocess.Popen([sys.executable, "-B", "-c", program, str(directory)],
                    cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    shell=False, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x00000004)
                identity = causal_resources.process_identity(process)
                self.assertFalse(ready.exists())
                job.assign(process)
                self.assertFalse(ready.exists())
                causal_resources.resume_suspended(process)
                until = time.monotonic() + 2
                while not ready.exists() and process.poll() is None and time.monotonic() < until:
                    time.sleep(0.02)
                self.assertTrue(ready.exists())
                stop.touch()
                self.assertEqual(process.wait(timeout=2), 0)
                self.assertEqual(causal_resources.process_identity(process), identity)
                memory = causal_resources.process_memory(process)
                self.assertGreater(memory["peak_working_set_bytes"], 0)
                self.assertLessEqual(memory["peak_working_set_bytes"], causal_resources.CHILD_LIMIT)
            finally:
                job.close()
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
