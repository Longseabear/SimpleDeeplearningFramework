from pathlib import Path


def resolve_project_path(project_root: Path, path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root / path
