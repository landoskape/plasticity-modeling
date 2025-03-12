from pathlib import Path


def root_dir():
    return Path(__file__).parent.parent


def config_dir():
    return root_dir() / "configs"


def get_config_path(filename: str):
    return config_dir() / filename


def data_dir(folder: str | None = None):
    directory = root_dir() / "data"
    if folder is not None:
        directory = directory / folder
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def results_dir(folder: str | None = None):
    directory = root_dir() / "results"
    if folder is not None:
        directory = directory / folder
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_figure_dir(folder=None):
    figure_dir = root_dir() / "media" / "figures"
    if folder:
        figure_dir = figure_dir / folder
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def get_natural_images_dir():
    return root_dir() / "data" / "IMAGES.mat"
