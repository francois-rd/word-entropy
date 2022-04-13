from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

from utils.pathing import (
    makepath,
    ensure_path,
    ExperimentPaths,
    EXPERIMENT_DIR,
    NEO_DATA_DIR,
    TIME_SERIES_DIR,
    PLOT_TS_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    EXISTING_FILE
)
from utils.timeline import TimelineConfig, Timeline
from utils.config import CommandConfigBase


class PlotTimeSeriesConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the PlotTimeSeries class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        input_dir: (type: Path-like, default: utils.pathing.TIME_SERIES_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read the time series data.

        surviving_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Path (relative to 'input_dir') of the surviving new word time series
            file.

        dying_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'input_dir') of the dying new word time series
            file.

        existing_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'input_dir') of the existing word time series
            file.

        neo_dir: (type: Path-like, default: utils.pathing.NEO_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the new and existing words.

        surviving_neo_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Path (relative to 'neo_dir') to the detected surviving new words.

        dying_neo_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'neo_dir') to the detected dying new words file.

        existing_neo_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'neo_dir') to the randomly-sampled existing words
            file.

        output_dir: (type: Path-like, default: utils.pathing.PLOT_TS_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        num_anecdotes: (type: int, default: 5)
            Number of anecdotal words to randomly sample for plotting.

        drop_last: (type: bool, default: True)
            Whether to drop the last time slice or not.

        major_x_ticks: (type: int, default: 0)
            If positive, the value to set for the major xticks in matplotlib.
            Otherwise, leave xticks to the default layout.

        legend_y_anchor: (type: dict[str, float], default: {})
            For the mean plots, y-value of the legend bounding box anchor for
            each type of time series. If missing, value set to 0.0 by default.

        plot_std: (type: bool, default: True)
            Whether to plot the standard deviations in the mean plots or not.

        quantile: (type: float, default: 0.25)
            The quantiles at which to bin the data for the quantile plots.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', TIME_SERIES_DIR)
        self.surviving_file = kwargs.pop('surviving_file', SURVIVING_FILE)
        self.dying_file = kwargs.pop('dying_file', DYING_FILE)
        self.existing_file = kwargs.pop('existing_file', EXISTING_FILE)
        self.neo_dir = kwargs.pop('neo_dir', NEO_DATA_DIR)
        self.surviving_neo_file = kwargs.pop(
            'surviving_neo_file', SURVIVING_FILE)
        self.dying_neo_file = kwargs.pop('dying_neo_file', DYING_FILE)
        self.existing_neo_file = kwargs.pop('existing_neo_file', EXISTING_FILE)
        self.output_dir = kwargs.pop('output_dir', PLOT_TS_DIR)
        self.num_anecdotes = kwargs.pop('num_anecdotes', 5)
        self.drop_last = kwargs.pop('drop_last', True)
        self.major_x_ticks = kwargs.pop('major_x_ticks', 0)
        self.legend_y_anchor = kwargs.pop('legend_y_anchor', {})
        self.plot_std = kwargs.pop('plot_std', True)
        self.quantile = kwargs.pop('quantile', 0.25)
        self.timeline_config = kwargs.pop('timeline_config', {})
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            time_series_dir=self.input_dir,
            neo_data_dir=self.neo_dir,
            plot_ts_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.time_series_dir
        self.surviving_file = makepath(self.input_dir, self.surviving_file)
        self.dying_file = makepath(self.input_dir, self.dying_file)
        self.existing_file = makepath(self.input_dir, self.existing_file)
        self.neo_dir = paths.neo_data_dir
        self.surviving_neo_file = makepath(
            self.neo_dir, self.surviving_neo_file)
        self.dying_neo_file = makepath(self.neo_dir, self.dying_neo_file)
        self.existing_neo_file = makepath(self.neo_dir, self.existing_neo_file)
        self.output_dir = paths.plot_ts_dir
        return self


class PlotTimeSeries:
    def __init__(self, config: PlotTimeSeriesConfig):
        """
        Plots the computed time series representation of new word volatility in
        various ways.

        :param config: see PlotTimeSeriesConfig for details
        """
        self.config = config
        tl_config = TimelineConfig(**self.config.timeline_config)
        self.max_time_slice = Timeline(tl_config).slice_of(tl_config.end)
        self.max_time_slice -= tl_config.early
        if config.drop_last:
            self.max_time_slice -= 1
        self.slice_size = tl_config.slice_size
        self.style = None

    def run(self) -> None:
        styles = ['seaborn-colorblind', 'seaborn-deep', 'dark_background']
        for style in styles:
            self.style = style   # Can't get this programmatically from context.
            with plt.style.context(style):
                surv = self._do_run("Surviving", self.config.surviving_file,
                                    self.config.surviving_neo_file)
                dying = self._do_run("Dying", self.config.dying_file,
                                     self.config.dying_neo_file)
                existing = self._do_run("Existing", self.config.existing_file,
                                        self.config.existing_neo_file)
                self._plot(surv[0], dying[0], existing[0])
                self._plot_quantiles(surv[1], dying[1], existing[1])

    def _do_run(self, word_type, input_path, neo_path):
        with open(input_path, 'rb') as file:
            all_time_series_by_word = pickle.load(file)
        self._maybe_drop_last(all_time_series_by_word)
        self._plot_anecdotes(word_type, all_time_series_by_word)
        qs = self._split_quantiles(word_type, all_time_series_by_word, neo_path)
        with_word_type = word_type, self._swap_keys(all_time_series_by_word)
        self._plot(with_word_type)
        self._plot_quantiles(qs)
        return with_word_type, qs

    def _maybe_drop_last(self, all_time_series_by_word):
        if self.config.drop_last:
            for time_series_for_word in all_time_series_by_word.values():
                for time_series in time_series_for_word.values():
                    if len(time_series) > self.max_time_slice:
                        del time_series[-1]

    def _plot_anecdotes(self, word_type, all_time_series_by_word):
        anecdotes = {k: all_time_series_by_word[k] for k in random.sample(
            list(all_time_series_by_word), self.config.num_anecdotes)}
        for ts_type in ['User', 'Subreddit']:
            filename = f"anecdotal-{word_type}-{ts_type.lower()}-"

            # All time series plotted in one graph.
            plt.figure()
            for word, time_series in anecdotes.items():
                ts = time_series[ts_type.lower()]
                plt.plot(np.arange(len(ts)), ts, label=word)
            self._finalize_plot(
                title=f"{ts_type} Entropy Time Series for Randomly-Selected "
                      f"{word_type} Words",
                filename=filename + "all.pdf"
            )

            # Each time series in its own graph.
            for word, time_series in anecdotes.items():
                plt.figure()
                ts = time_series[ts_type.lower()]
                plt.plot(np.arange(len(ts)), ts)
                self._finalize_plot(
                    title=f"{ts_type} Entropy Time Series for '{word}' "
                          f"({word_type})",
                    filename=filename + f"{word}.pdf",
                    legend=False
                )

    @staticmethod
    def _swap_keys(all_time_series_by_word):
        swapped = {}
        for time_series_for_word in all_time_series_by_word.values():
            for time_series_name, time_series in time_series_for_word.items():
                swapped.setdefault(time_series_name, []).append(time_series)
        return swapped

    def _finalize_plot(self, *, title, filename, legend=True, leg_params=None):
        plt.xticks(np.arange(self.max_time_slice + 1))
        if self.config.major_x_ticks > 0:
            ax = plt.gca().xaxis
            ax.set_major_locator(MultipleLocator(self.config.major_x_ticks))
            ax.set_minor_locator(MultipleLocator(1))
        plt.xlabel(f"Time Index ({self.slice_size}s since first appearance)")
        plt.ylabel("Normalized Entropy")
        plt.title(title)
        if legend:
            leg_params = leg_params or {}
            plt.legend(**leg_params)
        plt.tight_layout()
        sub_dir = ensure_path(makepath(self.config.output_dir, self.style))
        plt.savefig(makepath(sub_dir, filename))
        plt.close()

    def _plot(self, *args):
        for ts_type in ['User', 'Subreddit']:
            plt.figure()
            for word_type, swapped_ts in args:
                # Variable length input, so can't vectorize easily.
                ts_list = swapped_ts[ts_type.lower()]
                index, means, stds = 0, [], []
                while True:
                    pool = []
                    for ts in ts_list:
                        if index < len(ts):
                            pool.append(ts[index])
                    if not pool:
                        break
                    means.append(np.mean(pool))
                    stds.append(np.std(pool))
                    index += 1
                means = np.array(means)
                stds = np.array(stds)

                # Plot means, stds, and linear regression line.
                x = np.arange(len(means))
                color = plt.plot(x, means, label=word_type)[0].get_color()
                if self.config.plot_std:
                    plt.fill_between(x, means - stds, means + stds,
                                     color=color, alpha=0.2)
                else:
                    y = [max(means + stds), min(means - stds)]
                    plt.scatter([1, 2], y, color='k', alpha=0)
                poly1d_fn = np.poly1d(np.polyfit(x, means, 1))
                plt.plot(x, poly1d_fn(x), color=color, linestyle='dashed')
            filename = f"{'-'.join(wt for wt, _ in args)}-{ts_type.lower()}.pdf"
            y0 = self.config.legend_y_anchor.get(ts_type.lower(), 0.0)
            self._finalize_plot(
                title=f"{ts_type}",
                filename=filename,
                legend=len(args) > 1,
                leg_params={'loc': 'center right', 'bbox_to_anchor': (1.0, y0)}
            )

    def _split_quantiles(self, word_type, all_time_series_by_word, neo_path):
        with open(neo_path, 'rb') as file:
            usage_dict = pickle.load(file)
        counts_by_word = {w: len(usage[2]) for w, usage in usage_dict.items()}
        count_sorted = sorted(counts_by_word.items(), key=lambda item: item[1])
        words, counts = zip(*count_sorted)
        quantiles = np.arange(0, 1, self.config.quantile)
        bins = np.quantile(counts, quantiles)
        inds = np.digitize(counts, bins) - 1
        splits = []
        for b, q in enumerate(quantiles + self.config.quantile):
            word_bin = [w for w, i in zip(words, inds) if i == b]
            ts_subset = {k: v for k, v in all_time_series_by_word.items()
                         if k in word_bin}
            splits.append((q, self._swap_keys(ts_subset)))
        return word_type, splits

    def _plot_quantiles(self, *args):
        for ts_type in ['User', 'Subreddit']:
            plt.figure()
            for word_type, splits in args:
                for q, swapped_ts in splits:
                    # Variable length input, so can't vectorize easily.
                    ts_list = swapped_ts[ts_type.lower()]
                    index, means, stds = 0, [], []
                    while True:
                        pool = []
                        for ts in ts_list:
                            if index < len(ts):
                                pool.append(ts[index])
                        if not pool:
                            break
                        means.append(np.mean(pool))
                        stds.append(np.std(pool))
                        index += 1
                    means = np.array(means)
                    stds = np.array(stds)

                    # Plot means, stds, and linear regression line.
                    x = np.arange(len(means))
                    label = f"{word_type} ({q})"
                    color = plt.plot(x, means, label=label)[0].get_color()
                    if self.config.plot_std:
                        plt.fill_between(x, means - stds, means + stds,
                                         color=color, alpha=0.2)
                    else:
                        y = [max(means + stds), min(means - stds)]
                        plt.scatter([1, 2], y, color='k', alpha=0)
                    poly1d_fn = np.poly1d(np.polyfit(x, means, 1))
                    plt.plot(x, poly1d_fn(x), color=color, linestyle='dashed')
            filename = f"quant-{'-'.join(wt for wt, _ in args)}-{ts_type}.pdf"
            self._finalize_plot(title=f"{ts_type}", filename=filename,
                                leg_params={'loc': 'lower right'})
