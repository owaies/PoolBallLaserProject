from pathlib import Path

import pytest

from backend.app.api.detect import resolve_dataset_folder
from backend.app.core.config import settings


def test_resolves_relative_path_inside_dataset_root(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    target = project_root / "datasets" / "raw"
    target.mkdir(parents=True)
    monkeypatch.setattr(settings, "BASE_DIR", project_root)

    assert resolve_dataset_folder("datasets/raw") == target.resolve()


def test_rejects_absolute_path_outside_dataset_root(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "datasets").mkdir(parents=True)
    outside = tmp_path / "private"
    outside.mkdir()
    monkeypatch.setattr(settings, "BASE_DIR", project_root)

    with pytest.raises(ValueError, match="inside the datasets directory"):
        resolve_dataset_folder(str(outside))


def test_rejects_symlink_that_escapes_dataset_root(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    datasets = project_root / "datasets"
    datasets.mkdir(parents=True)
    outside = tmp_path / "private"
    outside.mkdir()
    (datasets / "external").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(settings, "BASE_DIR", project_root)

    with pytest.raises(ValueError, match="inside the datasets directory"):
        resolve_dataset_folder("datasets/external")


def test_rejects_files_instead_of_directories(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    datasets = project_root / "datasets"
    datasets.mkdir(parents=True)
    file_path = datasets / "not-a-folder.txt"
    file_path.write_text("not a folder", encoding="utf-8")
    monkeypatch.setattr(settings, "BASE_DIR", project_root)

    with pytest.raises(ValueError, match="must be a directory"):
        resolve_dataset_folder("datasets/not-a-folder.txt")
