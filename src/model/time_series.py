import numpy as np
import pickle

from utils.pathing import makepath, DIST_DIR, TIME_SERIES_DIR
from utils.pathing import SURVIVING_FILE, DYING_FILE
from utils.misc import warn_not_empty


class TimeSeriesConfig:
    def __init__(self, **kwargs):
        """
        Configs for the TimeSeries class. Accepted kwargs are:

        input_dir: (type: Path-like, default: utils.pathing.DIST_DIR)
            Root directory from which to read the word frequency distributions.

        surviving_input_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Name of the surviving new word distributions output file.

        dying_input_path: (type: str, default: utils.pathing.DYING_FILE)
            Name of the dying new word distributions output file.

        output_dir: (type: Path-like, default: utils.pathing.TIME_SERIES_DIR)
            Root directory in which to store all the output files.

        surviving_output_file: (type: str, default:
                utils.pathing.SURVIVING_FILE)
            Name of the surviving new word entropy time series output file.

        dying_output_file: (type: str, default: utils.pathing.DYING_FILE)
            Name of the surviving new word entropy time series output file.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        # NOTE: this assumes full path to files, not just filenames.
        self.input_dir = kwargs.pop('input_dir', str(DIST_DIR))
        self.surviving_input_file = kwargs.pop(
            'surviving_input_file', SURVIVING_FILE)
        self.dying_input_file = kwargs.pop('dying_input_file', DYING_FILE)
        self.output_dir = kwargs.pop('output_dir', str(TIME_SERIES_DIR))
        self.surviving_output_file = kwargs.pop(
            'surviving_output_file', SURVIVING_FILE)
        self.dying_output_file = kwargs.pop('dying_output_file', DYING_FILE)
        warn_not_empty(kwargs)


class TimeSeries:
    def __init__(self, config: TimeSeriesConfig):
        """
        Computes user and subreddit entropy time series representation from the
        word frequency distributions for all novel words over the time slice
        specified by a Timeline.

        :param config: see TimeSeriesConfig for details
        """
        self.config = config

    def run(self) -> None:
        config = self.config
        self._do_run(config.surviving_input_file, config.surviving_output_file)
        self._do_run(config.dying_input_file, config.dying_output_file)

    def _do_run(self, input_file, output_file):
        with open(makepath(self.config.input_dir, input_file), 'rb') as file:
            dists = pickle.load(file)
        time_series = {word: self._process(dists[word]) for word in dists}
        with open(makepath(self.config.output_dir, output_file), 'wb') as file:
            pickle.dump(time_series, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _process(time_slices):
        slice_indices = [int(index) for index in time_slices.keys()]
        offset = min(slice_indices)
        all_slices = [0.0] * (max(slice_indices) - offset + 1)
        all_time_series = {'user': all_slices, 'subreddit': all_slices}
        for index, all_dists in time_slices.items():
            for dist_name, dist in all_dists.items():
                freqs = np.array(list(dist.values()))
                total = np.sum(freqs)
                entropy = np.log2(total) - freqs.dot(np.log2(freqs)) / total
                all_time_series[dist_name][int(index) - offset] = entropy
        return all_time_series
