from pathlib import Path
from freezedry import freezedry


def root_dir() -> Path:
    return Path(__file__).parent.parent


def config_dir() -> Path:
    return root_dir() / "configs"


def get_config_path(filename: str) -> Path:
    return config_dir() / filename


def data_dir(folder: str | None = None) -> Path:
    directory = root_dir() / "data"
    if folder is not None:
        directory = directory / folder
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def results_dir(folder: str | None = None) -> Path:
    directory = root_dir() / "results"
    if folder is not None:
        directory = directory / folder
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_figure_dir(folder: str | None = None) -> Path:
    figure_dir = root_dir() / "media" / "figures"
    if folder:
        figure_dir = figure_dir / folder
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def get_natural_images_dir() -> Path:
    return root_dir() / "data" / "IMAGES.mat"


def save_repo_snapshot(output_path: Path, verbose: bool = True) -> None:
    freezedry(directory_path=root_dir(), output_path=output_path, ignore_git=True, use_gitignore=True, verbose=verbose)
