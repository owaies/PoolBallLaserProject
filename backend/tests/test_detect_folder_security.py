from pathlib import Path
import tempfile
import unittest

from backend.app.services.path_security import resolve_dataset_folder


class DatasetFolderPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "project"
        self.datasets = self.root / "datasets"
        self.datasets.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resolves_relative_path_inside_dataset_root(self) -> None:
        target = self.datasets / "raw"
        target.mkdir()
        self.assertEqual(
            resolve_dataset_folder("datasets/raw", self.root, self.datasets),
            target.resolve(),
        )

    def test_rejects_absolute_path_outside_dataset_root(self) -> None:
        outside = Path(self.temp_dir.name) / "private"
        outside.mkdir()
        with self.assertRaisesRegex(ValueError, "inside the datasets directory"):
            resolve_dataset_folder(str(outside), self.root, self.datasets)

    def test_rejects_symlink_that_escapes_dataset_root(self) -> None:
        outside = Path(self.temp_dir.name) / "private"
        outside.mkdir()
        (self.datasets / "external").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside the datasets directory"):
            resolve_dataset_folder("datasets/external", self.root, self.datasets)

    def test_rejects_files_instead_of_directories(self) -> None:
        file_path = self.datasets / "not-a-folder.txt"
        file_path.write_text("not a folder", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be a directory"):
            resolve_dataset_folder("datasets/not-a-folder.txt", self.root, self.datasets)


if __name__ == "__main__":
    unittest.main()
