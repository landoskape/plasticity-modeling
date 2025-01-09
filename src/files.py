from pathlib import Path


def root_dir():
    return Path(__file__).parent.parent


def get_figure_dir(folder=None):
    figure_dir = root_dir() / "media" / "figures"
    if folder:
        figure_dir = figure_dir / folder
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def get_natural_images_dir():
    return root_dir() / "data" / "IMAGES.mat"
