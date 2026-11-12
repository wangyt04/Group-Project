import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_dataset_package import TEST_INDICES, TRAIN_INDICES, audit

EXPECTED = {
    "camera_angle_x": 1.586227683125316,
    "fl_x": 664.6633833646774,
    "fl_y": 664.6633672714233,
    "cx": 675.0,
    "cy": 540.0,
    "w": 1350,
    "h": 1080,
}


def write_transforms(path, indices):
    frames = [
        {
            "file_path": str(i),
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        for i in indices
    ]
    path.write_text(json.dumps({**EXPECTED, "frames": frames}), encoding="utf-8")


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "VRCaps_lumen_100"
        (self.root / "train_views").mkdir(parents=True)
        (self.root / "depth_train").mkdir()
        (self.root / "test_views_1").mkdir()
        for i in TRAIN_INDICES:
            (self.root / "train_views" / f"{i:04d}.png").write_bytes(b"")
            (self.root / "depth_train" / f"{i:04d}.tiff").write_bytes(b"")
        for i in TEST_INDICES:
            (self.root / "test_views_1" / f"{i:04d}.png").write_bytes(b"")
        write_transforms(self.root / "transforms.json", TRAIN_INDICES)
        write_transforms(self.root / "transforms_test.json", TEST_INDICES)
        (self.root / "prendo_config.json").write_text(
            json.dumps({"depth_scale": 1.0, "test_dir": "test_views_1"}), encoding="utf-8"
        )
        (self.root / "init_point_cloud.ply").write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 300000\n"
            b"property float x\nproperty float y\nproperty float z\nend_header\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_package_passes(self):
        report = audit(self.root)
        self.assertTrue(report["ok"], report["errors"])

    def test_missing_depth_fails(self):
        (self.root / "depth_train" / "0000.tiff").unlink()
        report = audit(self.root)
        self.assertFalse(report["ok"])
        self.assertTrue(any("depth_train" in e for e in report["errors"]))

    def test_wrong_depth_scale_fails(self):
        (self.root / "prendo_config.json").write_text(
            json.dumps({"depth_scale": 1.4, "test_dir": "test_views_1"}), encoding="utf-8"
        )
        report = audit(self.root)
        self.assertFalse(report["ok"])
        self.assertTrue(any("depth_scale" in e for e in report["errors"]))

    def test_c3vd_path_fails(self):
        c3vd_root = Path(self.tmp.name) / "C3VD_pkg"
        self.root.rename(c3vd_root)
        report = audit(c3vd_root)
        self.assertFalse(report["ok"])
        self.assertTrue(any("C3VD" in e for e in report["errors"]))

    def test_wrong_test_index_fails(self):
        (self.root / "test_views_1" / "0095.png").rename(
            self.root / "test_views_1" / "0096.png"
        )
        report = audit(self.root)
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
