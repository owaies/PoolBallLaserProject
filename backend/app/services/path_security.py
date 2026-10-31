from pathlib import Path


def resolve_dataset_folder(folder_path: str, base_dir: Path, datasets_dir: Path) -> Path:
    """Resolve a requested folder and ensure it stays inside the dataset root."""
    if not isinstance(folder_path, str) or not folder_path.strip():
        raise ValueError("A detection folder is required.")

    requested = Path(folder_path).expanduser()
    if not requested.is_absolute():
        requested = base_dir / requested

    target_dir = requested.resolve(strict=True)
    datasets_root = datasets_dir.resolve(strict=True)
    if not target_dir.is_dir():
        raise ValueError("Detection path must be a directory.")
    if not target_dir.is_relative_to(datasets_root):
        raise ValueError("Detection folder must be inside the datasets directory.")
    return target_dir
