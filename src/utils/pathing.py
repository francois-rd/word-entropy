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


# Root-level paths.
PROJECT_ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
EXPERIMENTS_ROOT_DIR = makepath(PROJECT_ROOT_DIR, "experiments")
EXPERIMENT_DIR = "main"

# High-level paths.
DATA_DIR = "data"
MODEL_DIR = "model"
RESULTS_DIR = "results"

# Data-specific paths.
CACHE_DIR = makepath(DATA_DIR, "cache")
RAW_DATA_DIR = makepath(DATA_DIR, "raw")
PREPROC_DATA_DIR = makepath(DATA_DIR, "preprocessed")
USAGES_DATA_DIR = makepath(DATA_DIR, "usages")
NEO_DATA_DIR = makepath(DATA_DIR, "neologisms")

# Model-specific paths.
DIST_DIR = makepath(MODEL_DIR, "distributions")
TIME_SERIES_DIR = makepath(MODEL_DIR, "time_series")

# Recurring files.
USAGE_DICT_FILE = "usage_dict.pickle"
ID_MAP_FILE = "id_map.pickle"
SURVIVING_FILE = "surviving.pickle"
DYING_FILE = "dying.pickle"


class ExperimentPaths:
    def __init__(
            self,
            experiment_dir=EXPERIMENT_DIR,
            data_dir=DATA_DIR,
            model_dir=MODEL_DIR,
            results_dir=RESULTS_DIR,
            cache_dir=CACHE_DIR,
            raw_data_dir=RAW_DATA_DIR,
            preproc_data_dir=PREPROC_DATA_DIR,
            usages_data_dir=USAGES_DATA_DIR,
            neo_data_dir=NEO_DATA_DIR,
            dist_dir=DIST_DIR,
            time_series_dir=TIME_SERIES_DIR
    ):
        """
        Utilities for paths relating to the current experiment.

        'experiment_dir' is relative to 'utils.pathing.EXPERIMENTS_ROOT_DIR'
        unless given as an absolute path.

        All other '*_dir' are relative to 'experiment_dir' unless given as
        absolute paths.
        """
        self.experiment_dir = self._process(
            experiment_dir, root=EXPERIMENTS_ROOT_DIR)
        self.data_dir = self._process(data_dir)
        self.model_dir = self._process(model_dir)
        self.results_dir = self._process(results_dir)
        self.cache_dir = self._process(cache_dir)
        self.raw_data_dir = self._process(raw_data_dir)
        self.preproc_data_dir = self._process(preproc_data_dir)
        self.usages_data_dir = self._process(usages_data_dir)
        self.neo_data_dir = self._process(neo_data_dir)
        self.dist_dir = self._process(dist_dir)
        self.time_series_dir = self._process(time_series_dir)

    def _process(self, path, root=None):
        root = root or self.experiment_dir
        if pathlib.Path(path).is_absolute():
            return str(ensure_path(path))
        return ensure_path(makepath(root, path))
