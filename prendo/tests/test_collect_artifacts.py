import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.collect_artifacts import collect


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.run_dir = base / "run"
        self.data_dir = base / "data"
        (self.run_dir / "point_cloud/iteration_30000").mkdir(parents=True)
        (self.run_dir / "test/ours_30000/renders").mkdir(parents=True)
        (self.run_dir / "point_cloud/iteration_30000/point_cloud.ply").write_bytes(b"ply")
        (self.run_dir / "chkpnt30000.pth").write_bytes(b"ckpt")
        (self.run_dir / "results.txt").write_text(
            "[ITER 30000] Evaluating test avg: L1 0.1000 PSNR 20.5000 LPIPS 0.2000 SSIM 0.7000 MSSIM 0.8000\n",
            encoding="utf-8",
        )
        (self.run_dir / "test/ours_30000/renders/0007.png").write_bytes(b"png")
        (self.run_dir / "job_info.json").write_text(
            json.dumps({"pbs_jobid": "123", "walltime_seconds": 42, "resolution": 1, "iterations": 30000}),
            encoding="utf-8",
        )
        (self.run_dir / "nvidia_smi.txt").write_text(
            "full nvidia-smi output\n", encoding="utf-8"
        )
        (self.run_dir / "gpu_info.txt").write_text(
            "NVIDIA L40S, 46068\n", encoding="utf-8"
        )
        (self.run_dir / "gpu_mem_samples.txt").write_text("1024\n30720\n2048\n", encoding="utf-8")
        (self.run_dir / "command.txt").write_text("python train_viewdir.py\n", encoding="utf-8")
        self.data_dir.mkdir()
        for name in ("transforms.json", "transforms_test.json", "prendo_config.json", "init_point_cloud.ply"):
            (self.data_dir / name).write_bytes(name.encode())
        self.patch = base / "patch.diff"
        self.patch.write_bytes(b"patch")

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_fields_and_hashes(self):
        out = Path(self.tmp.name) / "run_manifest.json"
        manifest = collect(self.run_dir, self.data_dir, self.patch, out)
        saved = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(saved["run_id"], "run")
        self.assertEqual(saved["metrics"]["psnr"], 20.5)
        self.assertEqual(saved["hardware"]["gpu"], "NVIDIA L40S")
        self.assertAlmostEqual(saved["peak_vram_gb"], 30.0, places=3)
        self.assertEqual(manifest["reader_patch_sha256"], hashlib.sha256(b"patch").hexdigest())
        paths = {o["path"] for o in manifest["outputs"]}
        self.assertIn("point_cloud/iteration_30000/point_cloud.ply", paths)
        self.assertIn("test/ours_30000/renders/0007.png", paths)
        self.assertEqual(
            manifest["dataset"]["transforms.json"],
            hashlib.sha256(b"transforms.json").hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
