import numpy as np
import pickle

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    DIST_DIR,
    TIME_SERIES_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    EXISTING_FILE
)
from utils.config import CommandConfigBase


class TimeSeriesConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the TimeSeries class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        input_dir: (type: Path-like, default: utils.pathing.DIST_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read the word frequency distributions.

        surviving_input_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Path (relative to 'input_dir') of the surviving new word
            distributions file.

        dying_input_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'input_dir') of the dying new word distributions
            file.

        existing_input_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'input_dir') of the randomly-sampled existing
            word distributions file.

        output_dir: (type: Path-like, default: utils.pathing.TIME_SERIES_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        surviving_output_file: (type: str, default:
                utils.pathing.SURVIVING_FILE)
            Path (relative to 'output_dir') of the surviving new word entropy
            time series output file.

        dying_output_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'output_dir') of the dying new word entropy time
            series output file.

        existing_output_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'output_dir') of the existing word entropy time
            series output file.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', DIST_DIR)
        self.surviving_input_file = kwargs.pop(
            'surviving_input_file', SURVIVING_FILE)
        self.dying_input_file = kwargs.pop('dying_input_file', DYING_FILE)
        self.existing_input_file = kwargs.pop(
            'existing_input_file', EXISTING_FILE)
        self.output_dir = kwargs.pop('output_dir', TIME_SERIES_DIR)
        self.surviving_output_file = kwargs.pop(
            'surviving_output_file', SURVIVING_FILE)
        self.dying_output_file = kwargs.pop('dying_output_file', DYING_FILE)
        self.existing_output_file = kwargs.pop(
            'existing_output_file', EXISTING_FILE)
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            dist_dir=self.input_dir,
            time_series_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.dist_dir
        self.surviving_input_file = makepath(
            self.input_dir, self.surviving_input_file)
        self.dying_input_file = makepath(self.input_dir, self.dying_input_file)
        self.existing_input_file = makepath(
            self.input_dir, self.existing_input_file)
        self.output_dir = paths.time_series_dir
        self.surviving_output_file = makepath(
            self.output_dir, self.surviving_output_file)
        self.dying_output_file = makepath(
            self.output_dir, self.dying_output_file)
        self.existing_output_file = makepath(
            self.output_dir, self.existing_output_file)
        return self


class TimeSeries:
    def __init__(self, config: TimeSeriesConfig):
        """
        Computes user and subreddit entropy time series representation from the
        word frequency distributions for all words over the time slice specified
        by a Timeline.

        :param config: see TimeSeriesConfig for details
        """
        self.config = config

    def run(self) -> None:
        config = self.config
        self._do_run(config.surviving_input_file, config.surviving_output_file)
        self._do_run(config.dying_input_file, config.dying_output_file)
        self._do_run(config.existing_input_file, config.existing_output_file)

    def _do_run(self, input_file, output_file):
        with open(input_file, 'rb') as file:
            dists = pickle.load(file)
        time_series = {word: self._process(dists[word]) for word in dists}
        with open(output_file, 'wb') as file:
            pickle.dump(time_series, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _process(time_slices):
        slice_indices = [int(index) for index in time_slices.keys()]
        offset = min(slice_indices)
        all_slices = [0.0] * (max(slice_indices) - offset + 1)
        all_time_series = {'user': all_slices, 'subreddit': all_slices.copy()}
        for index, all_dists in time_slices.items():
            for dist_name, dist in all_dists.items():
                freqs = np.array(list(dist.values()))
                if freqs.shape[0] == 1:
                    norm_entropy = 0.0
                else:
                    total = np.sum(freqs)
                    entropy = np.log2(total) - freqs.dot(np.log2(freqs)) / total
                    norm_entropy = entropy / np.log2(freqs.shape[0])
                all_time_series[dist_name][int(index) - offset] = norm_entropy
        return all_time_series
