import pandas as pd
import pickle
import os

from utils.pathing import makepath, PREPROC_DATA_DIR, USAGES_DATA_DIR
from utils.pathing import USAGE_DICT_FILE, ID_MAP_FILE
from utils.misc import warn_not_empty, ItemBlockMapper


class WordUsageFinderConfig:
    def __init__(self, **kwargs):
        """
        Configs for the WordUsageFinder class. Accepted kwargs are:

        input_dir: (type: Path-like, default: utils.pathing.PREPROC_DATA_DIR)
            Root directory from which to read all the preprocessed Reddit data.

        output_dir: (type: Path-like, default: utils.pathing.USAGES_DATA_DIR)
            Root directory in which to store all the output files.

        usage_file: (type: str, default: utils.pathing.USAGE_DICT_FILE)
            Name of the usage dictionary output file.

        map_file: (type: str, default: utils.pathing.ID_MAP_FILE)
            Name of the usage ID map output file.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        # NOTE: this assumes full path to files, not just filenames.
        self.input_dir = kwargs.pop('input_dir', str(PREPROC_DATA_DIR))
        self.output_dir = kwargs.pop('output_dir', str(USAGES_DATA_DIR))
        self.usage_file = kwargs.pop('usage_file', USAGE_DICT_FILE)
        self.map_file = kwargs.pop('map_file', ID_MAP_FILE)
        warn_not_empty(kwargs)


class WordUsageFinder:
    def __init__(self, config: WordUsageFinderConfig):
        """
        Creates a word-usage dictionary from the preprocessed Reddit data.

        :param config: see WordUsageFinderConfig for details
        """
        self.config = config
        self.word_usage = {}
        self.mapper = ItemBlockMapper()

    def run(self) -> None:
        for root, _, files in os.walk(self.config.input_dir):
            for file in files:
                self.mapper.new_block(file)
                df = pd.read_csv(makepath(root, file))
                for b, c in zip(df['body'], df['created_utc']):
                    self._process(b, c)
        usage_file = makepath(self.config.output_dir, self.config.usage_file)
        with open(usage_file, 'wb') as file:
            pickle.dump(self.word_usage, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.mapper.save(makepath(self.config.output_dir, self.config.map_file))

    def _process(self, body, created):
        comment_id = self.mapper.new_item_id()
        for word in body.split():
            usage = self.word_usage.setdefault(word, [float('inf'), 0, []])
            usage[0] = min(created, usage[0])  # First usage.
            usage[1] = max(created, usage[1])  # Last usage.
            usage[2].append(comment_id)