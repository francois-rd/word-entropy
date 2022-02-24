import pathlib
import os


def ensure_path(path):
    """
    Ensures every directory along the path exists.

    :param path: some path
    :return: the path
    """
    os.makedirs(path, exist_ok=True)
    return path


def makepath(root, *args, as_string=True):
    """
    Constructs a path out of the given arguments.

    :param root: the path root
    :param args: any subdirectories or file
    :param as_string: whether to return a pathlib.Path or a string
    :return: the resulting path
    """
    path = pathlib.Path(root)
    for arg in args:
        path = path / arg
    return str(path) if as_string else path


# High-level paths.
PROJECT_ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ensure_path(PROJECT_ROOT_DIR / "data")
RESULTS_DIR = ensure_path(PROJECT_ROOT_DIR / "results")
