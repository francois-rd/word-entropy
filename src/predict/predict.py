# (Optional) Figure out how to do prec, rec, f1, and acc for each one.
# For each metric, make a num_predictors x k table to report the metric.
#  In principle, would need cross validation for means/std, but oh well.
# See also:
#  https://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting
#  https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
#  *** Esp this one: https://scikit-learn.org/stable/model_selection.html
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    TIME_SERIES_DIR,
    PREDICT_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    EXISTING_FILE
)
from utils.timeline import TimelineConfig, Timeline
from utils.config import CommandConfigBase
from . import ALL_PREDICTORS


class PredictionConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the Prediction class. Accepted kwargs are:

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

        output_dir: (type: Path-like, default: utils.pathing.PREDICT_DIR)
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
        self.output_dir = kwargs.pop('output_dir', PREDICT_DIR)
        self.timeline_config = kwargs.pop('timeline_config', {})
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            time_series_dir=self.input_dir,
            predict_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.time_series_dir
        self.surviving_file = makepath(self.input_dir, self.surviving_file)
        self.dying_file = makepath(self.input_dir, self.dying_file)
        self.existing_file = makepath(self.input_dir, self.existing_file)
        self.output_dir = paths.predict_dir
        return self


class Prediction:
    def __init__(self, seed: int, config: PredictionConfig):
        """
        Predicts the word type (surviving, dying, existing) from their entropy

        :param: a random seed for reproducible predictor behaviour
        :param config: see PredictionConfig for details
        """
        self.seed = seed
        self.config = config
        tl_config = TimelineConfig(**self.config.timeline_config)
        self.max_time_slice = Timeline(tl_config).slice_of(tl_config.end)
        self.max_time_slice -= tl_config.early
        self.slice_size = tl_config.slice_size
        self.metrics = {'Acc': accuracy_score, 'Bal': balanced_accuracy_score}
        self.predictors = [pred(self.seed) for pred in ALL_PREDICTORS]

    def run(self) -> None:
        surviving = self._extract(0, self.config.surviving_file)
        #dying = self._extract(1, self.config.dying_file)
        dying = self._extract(1, self.config.surviving_file)  # TODO: Fix.
        existing = self._extract(2, self.config.existing_file)
        self._do_run(surviving, dying)
        self._do_run(surviving, dying, existing=existing)

    def _do_run(self, surviving, dying, existing=None):
        scores = self._score(surviving, dying, existing)
        cols = [(f"Time Index ({self.slice_size}s since first appearance)",
                 str(i)) for i in range(len(surviving))]
        cols = pd.MultiIndex.from_tuples(cols)
        for metric_name, metric_scores in scores.items():
            self._table(metric_name, metric_scores, cols, existing is not None)

    def _extract(self, label, input_path):
        with open(input_path, 'rb') as file:
            all_time_series_by_word = pickle.load(file)
        train_keys, test_keys = train_test_split(
            list(all_time_series_by_word.keys()),
            test_size=0.1,
            random_state=self.seed
        )
        X_and_y_by_k = []
        for k in range(int(self.max_time_slice * 0.75)):
            X_train = self._do_extract(train_keys, k, all_time_series_by_word)
            X_test = self._do_extract(test_keys, k, all_time_series_by_word)
            X_and_y_by_k.append({'X_train': X_train, 'X_test': X_test,
                                 'y_train': [label] * len(X_train),
                                 'y_test': [label] * len(X_test)})
        return X_and_y_by_k

    @staticmethod
    def _do_extract(X_keys, k, all_time_series_by_word):
        X = []
        for key in X_keys:
            features = []
            for time_series in all_time_series_by_word[key].values():
                if len(time_series) <= k:  # This is NOT off-by-one.
                    features = None
                    break
                features.extend(time_series[:k + 1])
            if features is not None:
                X.append(features)
        return X

    def _score(self, surviving, dying, existing=None):
        scores = {}
        if existing is None:
            existing = [{'X_train': [], 'X_test': [],
                         'y_train': [], 'y_test': []} for _ in surviving]
        for s, d, e in zip(surviving, dying, existing):
            X_train = s['X_train'] + d['X_train'] + e['X_train']
            X_test = s['X_test'] + d['X_test'] + e['X_test']
            y_train = s['y_train'] + d['y_train'] + e['y_train']
            y_test = s['y_test'] + d['y_test'] + e['y_test']
            for predictor in self.predictors:
                predictor.find_best(X_train, y_train)
                y_pred = predictor.predict(X_test)
                for name, metric in self.metrics.items():
                    score = metric(y_test, y_pred)
                    data = scores.setdefault(name, {})
                    data.setdefault(predictor.name, []).append(score)
        return scores

    def _table(self, metric_name, metric_scores, cols, has_existing):
        filename = f"{metric_name}{'-Existing' if has_existing else ''}.txt"
        df = pd.DataFrame.from_dict(metric_scores, orient='index', columns=cols)
        df.style.format(precision=2).applymap_index(
            lambda v: "textbf:--rwrap;", axis=1
        ).applymap_index(
            lambda v: "textbf:--rwrap;"
        ).to_latex(
            buf=makepath(self.config.output_dir, filename),
            column_format="l|" + "c" * len(cols),
            position="h",
            position_float="centering",
            hrules=True,
            multicol_align="c"
        )
