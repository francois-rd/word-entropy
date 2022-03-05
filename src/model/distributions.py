from collections import defaultdict
import pandas as pd
import pickle

from utils.pathing import (
    makepath,
    PREPROC_DATA_DIR,
    NEO_DATA_DIR,
    USAGES_DATA_DIR,
    DIST_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    ID_MAP_FILE
)
from utils.timeline import TimelineConfig, Timeline
from utils.misc import warn_not_empty, ItemBlockMapper


class DistributionsConfig:
    def __init__(self, **kwargs):
        """
        Configs for the Distributions class. Accepted kwargs are:

        preproc_dir: (type: Path-like, default: utils.pathing.PREPROC_DATA_DIR)
            Root directory from which to read all the preprocessed Reddit data.

        surviving_input_path: (type: str, default:
                utils.pathing.NEO_DATA_DIR + utils.pathing.SURVIVING_FILE)
            Path to the detected surviving new words input file.

        dying_input_path: (type: str, default:
                utils.pathing.NEO_DATA_DIR + utils.pathing.DYING_FILE)
            Path to the detected dying new words input file.

        map_path: (type: str, default:
                utils.pathing.USAGES_DATA_DIR + utils.pathing.ID_MAP_FILE)
            Path to the usage ID map input file.

        output_dir: (type: Path-like, default: utils.pathing.DIST_DIR)
            Root directory in which to store all the output files.

        surviving_output_file: (type: str, default:
                utils.pathing.SURVIVING_FILE)
            Name of the surviving new word distributions output file.

        dying_output_file: (type: str, default: utils.pathing.DYING_FILE)
            Name of the dying new word distributions output file.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        # NOTE: this assumes full path to files, not just filenames.
        self.preproc_dir = kwargs.pop('preproc_dir', str(PREPROC_DATA_DIR))
        self.surviving_input_path = kwargs.pop(
            'surviving_input_path', makepath(str(NEO_DATA_DIR), SURVIVING_FILE))
        self.dying_input_path = kwargs.pop(
            'dying_input_path', makepath(str(NEO_DATA_DIR), DYING_FILE))
        self.map_path = kwargs.pop(
            'map_path', makepath(str(USAGES_DATA_DIR), ID_MAP_FILE))
        self.output_dir = kwargs.pop('output_dir', str(DIST_DIR))
        self.surviving_output_file = kwargs.pop(
            'surviving_output_file', SURVIVING_FILE)
        self.dying_output_file = kwargs.pop('dying_output_file', DYING_FILE)
        self.timeline_config = kwargs.pop('timeline_config', {})
        warn_not_empty(kwargs)


class Distributions:
    def __init__(self, config: DistributionsConfig):
        """
        Computes user and subreddit word frequency distributions for all novel
        words and for each time slice as specified by a Timeline.

        :param config: see DistributionsConfig for details
        """
        self.config = config
        self.timeline = Timeline(TimelineConfig(**self.config.timeline_config))
        self.mapper = ItemBlockMapper.load(self.config.map_path)

    def run(self) -> None:
        config = self.config
        self._do_run(config.surviving_input_path, config.surviving_output_file)
        self._do_run(config.dying_input_path, config.dying_output_file)

    def _do_run(self, input_path, output_file):
        with open(input_path, 'rb') as file:
            usage_dict = pickle.load(file)
        file_map = self._make_file_map(usage_dict)
        dists = {}  # Can't use defaultdict because we need to pickle after.
        for file, row_map in file_map.items():
            self._process_file(file, row_map, dists)
        self._save(dists, output_file)

    def _make_file_map(self, usage_dict):
        file_map = defaultdict(lambda: defaultdict(list))
        for word, usage in usage_dict.items():
            for comment_id in usage[2]:
                file, row = self.mapper.get_block_id(comment_id)
                file_map[file][row].append(word)
        return file_map

    def _process_file(self, file, row_map, dists):
        df = pd.read_csv(makepath(self.config.preproc_dir, file))
        for row_id, word_list in row_map.items():
            row = df.iloc[[row_id]]
            author_fullname = row['author_fullname'].item()
            subreddit_id = row['subreddit_id'].item()
            for word in word_list:
                all_slices = dists.setdefault(word, {})
                time_slice = self.timeline.slice_of(row['created_utc'].item())
                all_dists = all_slices.setdefault(time_slice, {})
                user = all_dists.setdefault('user', {})
                user[author_fullname] = user.get(author_fullname, 0) + 1
                subreddit = all_dists.setdefault('subreddit', {})
                subreddit[subreddit_id] = subreddit.get(subreddit_id, 0) + 1

    def _save(self, dists, filename):
        with open(makepath(self.config.output_dir, filename), 'wb') as file:
            pickle.dump(dists, file, protocol=pickle.HIGHEST_PROTOCOL)
