from scipy.stats import spearmanr, ttest_ind_from_stats
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import logging
import pickle
import math

from utils.pathing import (
    makepath,
    ensure_path,
    ExperimentPaths,
    EXPERIMENT_DIR,
    TIME_SERIES_DIR,
    STATS_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    EXISTING_FILE
)
from utils.timeline import TimelineConfig, Timeline
from utils.config import CommandConfigBase


class PlotStatsConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the PlotStats class. Accepted kwargs are:

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

        output_dir: (type: Path-like, default: utils.pathing.STATS_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

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
        self.output_dir = kwargs.pop('output_dir', STATS_DIR)
        self.timeline_config = kwargs.pop('timeline_config', {})
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            time_series_dir=self.input_dir,
            stats_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.time_series_dir
        self.surviving_file = makepath(self.input_dir, self.surviving_file)
        self.dying_file = makepath(self.input_dir, self.dying_file)
        self.existing_file = makepath(self.input_dir, self.existing_file)
        self.output_dir = paths.stats_dir
        return self


class PlotStats:
    def __init__(self, config: PlotStatsConfig):
        """
        Plots the computed time series representation of new word volatility in
        various ways.

        :param config: see PlotStatsConfig for details
        """
        self.config = config
        tl_config = TimelineConfig(**self.config.timeline_config)
        self.max_time_slice = Timeline(tl_config).slice_of(tl_config.end)
        self.max_time_slice -= tl_config.early
        self.slice_size = tl_config.slice_size
        self.style = None
        self.all_time_series_names = None

    def run(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        surv = self._do_run("Surviving", self.config.surviving_file)
        #dying = self._do_run("Dying", self.config.dying_file)
        existing = self._do_run("Existing", self.config.existing_file)
        #args = [surv, dying, existing]
        args = [surv, existing]
        self._table(*args)
        styles = ['seaborn-colorblind', 'seaborn-deep', 'dark_background']
        for style in styles:
            self.style = style  # Can't get this programmatically from context.
            with plt.style.context(style):
                for word_type in args:
                    self._plot(word_type)
                self._plot(*args)

    def _do_run(self, word_type, input_path):
        with open(input_path, 'rb') as file:
            all_time_series_by_word = pickle.load(file)
        rhos = self._spearman(all_time_series_by_word)
        with_word_type = word_type, self._stats(rhos)
        return with_word_type

    def _spearman(self, all_time_series_by_word):
        rhos, swapped = {}, self._swap_keys(all_time_series_by_word)
        if self.all_time_series_names is None:
            all_time_series_names = list([k.title() for k in swapped])
            self.all_time_series_names = all_time_series_names
        for time_series_name, time_series_list in swapped.items():
            rhos[time_series_name] = []
            for time_series in time_series_list:
                rhos_for_k = []
                for k in range(1, len(time_series)):
                    rho = spearmanr(np.arange(k + 1), time_series[:k + 1])[0]
                    if math.isnan(rho):
                        logging.warning("Treating NaN rho as 0.0")
                        rho = 0.0
                    rhos_for_k.append(rho)
                rhos[time_series_name].append(rhos_for_k)
        return rhos

    @staticmethod
    def _stats(rhos):
        # Variable length input, so can't vectorize easily.
        stats = {}
        for ts_name, time_series_list in rhos.items():
            index, means, stds, nobs = 0, [], [], []
            while True:
                pool = []
                for time_series in time_series_list:
                    if index < len(time_series):
                        pool.append(time_series[index])
                if not pool:
                    break
                means.append(np.mean(pool))
                stds.append(np.std(pool))
                nobs.append(len(pool))
                index += 1
            stats[ts_name] = [np.array(means), np.array(stds), np.array(nobs)]
        return stats

    @staticmethod
    def _swap_keys(all_time_series_by_word):
        swapped = {}
        for time_series_for_word in all_time_series_by_word.values():
            for time_series_name, time_series in time_series_for_word.items():
                swapped.setdefault(time_series_name, []).append(time_series)
        return swapped

    def _finalize_plot(self, *, ylabel, title, filename, legend=True):
        plt.xticks(np.arange(self.max_time_slice + 1))
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.xlabel(f"Time Index ({self.slice_size}s since first appearance)")
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend()
        sub_dir = ensure_path(makepath(self.config.output_dir, self.style))
        plt.savefig(makepath(sub_dir, filename))
        plt.close()

    def _plot(self, *args):
        for ts_type in self.all_time_series_names:
            title = f"{ts_type} Time Series"
            filename = f"{'-'.join(wt for wt, _ in args)}-{ts_type.lower()}.pdf"
            plt.figure()
            for word_type, stat_ts in args:
                means, stds, _ = stat_ts[ts_type.lower()]
                x = np.arange(1, len(means) + 1)
                color = plt.plot(x, means, label=word_type)[0].get_color()
                plt.fill_between(x, means - stds, means + stds,
                                 color=color, alpha=0.2)
                poly1d_fn = np.poly1d(np.polyfit(x, means, 1))
                plt.plot(x, poly1d_fn(x), color=color, linestyle='dashed')
            self._finalize_plot(ylabel="Spearman's Rho", title=title,
                                filename=filename, legend=len(args) > 1)

    def _table(self, *args):
        n_combs = len(list(itertools.combinations(range(len(args)), 2)))
        n_tests = (len(args) + n_combs) * len(self.all_time_series_names)
        n_tests *= self.max_time_slice - 1
        data, all_index = [], []
        args_dict = {k[0]: v for k, v in args}
        for ts_type in self.all_time_series_names:
            index = []
            for word_type, stats_ts in args:
                index.append(word_type[0])
                self._ttest(stats_ts[ts_type.lower()], n_tests, data)
            for a, b in list(itertools.combinations(index, 2)):
                index.append(f"{a} - {b}")
                self._ttest(args_dict[a][ts_type.lower()], n_tests, data,
                            second_stats_ts=args_dict[b][ts_type.lower()])
            all_index.extend([(ts_type, i) for i in index])
        filename = f"{'-'.join(wt for wt, _ in args)}.txt"
        cols = [(f"Time Index ({self.slice_size}s since first appearance)",
                str(i)) for i in range(1, self.max_time_slice)]
        all_index = pd.MultiIndex.from_tuples(all_index)
        cols = pd.MultiIndex.from_tuples(cols)
        self._save(pd.DataFrame(data, index=all_index, columns=cols), filename)

    @staticmethod
    def _ttest(stats_ts, n_tests, data, second_stats_ts=None):
        row_data = []
        if second_stats_ts is None:
            reps = len(stats_ts[0])
            second_stats_ts = [[0.0] * reps, [0.0] * reps, [2] * reps]
        for m1, s1, n1, m2, s2, n2 in zip(*stats_ts, *second_stats_ts):
            t = ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
            pval = t[1] * n_tests  # Bonferroni Correction.
            if pval < 0.001:
                star = "^{***}"
            elif pval < 0.01:
                star = "^{**}"
            elif pval < 0.05:
                star = "^{*}"
            else:
                star = ""
            row_data.append(f"{m1 - m2:.2f}{star}")
        data.append(row_data)

    def _save(self, df, filename):
        df.style.applymap_index(
            lambda v: "rotatebox:{90}--rwrap;", level=0
        ).applymap_index(
            lambda v: "multicolumn:{1}{c}--rwrap;", level=1, axis=1
        ).applymap_index(
            lambda v: "textbf:--rwrap;", axis=1
        ).applymap_index(
            lambda v: "textbf:--rwrap;"
        ).to_latex(
            buf=makepath(self.config.output_dir, filename),
            column_format="cc|" + "d{2.5}" * (self.max_time_slice - 1),
            position="h",
            position_float="centering",
            hrules=True,
            multirow_align="c",
            multicol_align="c"
        )
