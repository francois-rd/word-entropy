import pandas as pd
import pickle
import os

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    RAW_DATA_DIR,
    COUNT_DATA_DIR,
    COUNT_FILE
)
from utils.config import CommandConfigBase
import utils.data_management as dm


class RedditCounterConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the RedditCounter class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        input_dir: (type: Path-like, default: utils.pathing.RAW_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the downloaded Reddit data.

        output_dir: (type: Path-like, default: utils.pathing.COUNT_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all count output file.

        count_file: (type: str, default: utils.pathing.COUNT_FILE)
            Path (relative to 'output_dir') of the user and subreddit count
            output file.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', RAW_DATA_DIR)
        self.output_dir = kwargs.pop('output_dir', COUNT_DATA_DIR)
        self.count_file = kwargs.pop('count_file', COUNT_FILE)
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            raw_data_dir=self.input_dir,
            count_data_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.raw_data_dir
        self.output_dir = paths.count_data_dir
        self.count_file = makepath(self.output_dir, self.count_file)
        return self


class RedditCounter:
    def __init__(self, config: RedditCounterConfig):
        """
        Counts the number of words that each user comments and each subreddit
        contains.

        :param config: see RedditCounterConfig for details
        """
        self.config = config
        self.subreddits, self.users = {}, {}

    def run(self) -> None:
        for root, _, files in os.walk(self.config.input_dir):
            for file in files:
                subreddit_id = dm.parts(file)['subreddit_id']
                self.subreddits.setdefault(subreddit_id, 0)
                df = pd.read_csv(makepath(root, file))
                for a, b in zip(df['author_fullname'], df['body']):
                    count = len(b.split())
                    self.users.setdefault(a, 0)
                    self.users[a] += count
                    self.subreddits[subreddit_id] += count
        with open(self.config.count_file, 'wb') as f:
            obj = {'subreddits': self.subreddits, 'users': self.users}
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
