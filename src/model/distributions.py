from collections import defaultdict
import pandas as pd
import pickle

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    PREPROC_DATA_DIR,
    NEO_DATA_DIR,
    USAGES_DATA_DIR,
    DIST_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    ID_MAP_FILE
)
from utils.misc import ItemBlockMapper, parts_from_filename
from utils.timeline import TimelineConfig, Timeline
from utils.config import CommandConfigBase


class DistributionsConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the Distributions class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        preproc_dir: (type: Path-like, default: utils.pathing.PREPROC_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the preprocessed Reddit data.

        neo_dir: (type: Path-like, default: utils.pathing.NEO_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the detected (surviving and dying) new words.

        surviving_neo_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Path (relative to 'neo_dir') to the detected surviving new words.

        dying_neo_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'neo_dir') to the detected dying new words file.

        usages_dir: (type: Path-like, default: utils.pathing.USAGES_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read 'map_file'.

        map_file: (type: str, default: utils.pathing.ID_MAP_FILE)
            Path (relative to 'usages_dir') to the usage ID map file.

        output_dir: (type: Path-like, default: utils.pathing.DIST_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        surviving_output_file: (type: str, default:
                utils.pathing.SURVIVING_FILE)
            Path (relative to 'output_dir') of the surviving new word
            distributions output file.

        dying_output_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'output_dir') of the dying new word distributions
            output file.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.preproc_dir = kwargs.pop('preproc_dir', PREPROC_DATA_DIR)
        self.neo_dir = kwargs.pop('neo_dir', NEO_DATA_DIR)
        self.surviving_neo_file = kwargs.pop(
            'surviving_neo_file', SURVIVING_FILE)
        self.dying_neo_file = kwargs.pop('dying_neo_file', DYING_FILE)
        self.usages_dir = kwargs.pop('usages_dir', USAGES_DATA_DIR)
        self.map_file = kwargs.pop('map_file', ID_MAP_FILE)
        self.output_dir = kwargs.pop('output_dir', DIST_DIR)
        self.surviving_output_file = kwargs.pop(
            'surviving_output_file', SURVIVING_FILE)
        self.dying_output_file = kwargs.pop('dying_output_file', DYING_FILE)
        self.timeline_config = kwargs.pop('timeline_config', {})
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            preproc_data_dir=self.preproc_dir,
            neo_data_dir=self.neo_dir,
            usages_data_dir=self.usages_dir,
            dist_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.preproc_dir = paths.preproc_data_dir
        self.neo_dir = paths.neo_data_dir
        self.surviving_neo_file = makepath(
            self.neo_dir, self.surviving_neo_file)
        self.dying_neo_file = makepath(self.neo_dir, self.dying_neo_file)
        self.usages_dir = paths.usages_data_dir
        self.map_file = makepath(self.usages_dir, self.map_file)
        self.output_dir = paths.dist_dir
        self.surviving_output_file = makepath(
            self.output_dir, self.surviving_output_file)
        self.dying_output_file = makepath(
            self.output_dir, self.dying_output_file)
        return self


class Distributions:
    def __init__(self, config: DistributionsConfig):
        """
        Computes user and subreddit word frequency distributions for all novel
        words and for each time slice as specified by a Timeline.

        :param config: see DistributionsConfig for details
        """
        self.config = config
        self.timeline = Timeline(TimelineConfig(**self.config.timeline_config))
        self.mapper = ItemBlockMapper.load(self.config.map_file)

    def run(self) -> None:
        config = self.config
        self._do_run(config.surviving_neo_file, config.surviving_output_file)
        self._do_run(config.dying_neo_file, config.dying_output_file)

    def _do_run(self, input_path, output_path):
        with open(input_path, 'rb') as file:
            usage_dict = pickle.load(file)
        file_map = self._make_file_map(usage_dict)
        dists = {}  # Can't use defaultdict because we need to pickle after.
        for file, row_map in file_map.items():
            self._process_file(file, row_map, dists)
        with open(output_path, 'wb') as file:
            pickle.dump(dists, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _make_file_map(self, usage_dict):
        file_map = defaultdict(lambda: defaultdict(list))
        for word, usage in usage_dict.items():
            for comment_id in usage[2]:
                file, row = self.mapper.get_block_id(comment_id)
                file_map[file][row].append(word)
        return file_map

    def _process_file(self, file, row_map, dists):
        df = pd.read_csv(makepath(self.config.preproc_dir, file))
        subreddit_id = parts_from_filename(file)['subreddit_id']
        for row_id, word_list in row_map.items():
            row = df.iloc[[row_id]]
            author_fullname = row['author_fullname'].item()
            for word in word_list:
                all_slices = dists.setdefault(word, {})
                time_slice = self.timeline.slice_of(row['created_utc'].item())
                all_dists = all_slices.setdefault(time_slice, {})
                user = all_dists.setdefault('user', {})
                user[author_fullname] = user.get(author_fullname, 0) + 1
                subreddit = all_dists.setdefault('subreddit', {})
                subreddit[subreddit_id] = subreddit.get(subreddit_id, 0) + 1
