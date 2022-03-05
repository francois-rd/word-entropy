import pickle

from utils.pathing import makepath, USAGES_DATA_DIR, NEO_DATA_DIR
from utils.pathing import USAGE_DICT_FILE, SURVIVING_FILE, DYING_FILE
from utils.misc import warn_not_empty
from utils.timeline import TimelineConfig, Timeline


class BasicDetectorConfig:
    def __init__(self, **kwargs):
        """
        Configs for the BasicDetector class. Accepted kwargs are:

        input_dir: (type: Path-like, default: utils.pathing.USAGES_DATA_DIR)
            Root directory from which to read all the word usage data.

        output_dir: (type: Path-like, default: utils.pathing.NEO_DATA_DIR)
            Root directory in which to store all the output files.

        usage_file: (type: Path-like, default: utils.pathing.USAGE_DICT_FILE)
            Name of the usage dictionary input file.

        surviving_file: (type: Path-like, default: utils.pathing.SURVIVING_FILE)
            Name of the detected surviving new words output file.

        dying_file: (type: Path-like, default: utils.pathing.DYING_FILE)
            Name of the detected dying new words output file.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        # NOTE: this assumes full path to files, not just filenames.
        self.input_dir = kwargs.pop('input_dir', str(USAGES_DATA_DIR))
        self.output_dir = kwargs.pop('output_dir', str(NEO_DATA_DIR))
        self.usage_file = kwargs.pop('usage_file', USAGE_DICT_FILE)
        self.surviving_file = kwargs.pop('surviving_file', SURVIVING_FILE)
        self.dying_file = kwargs.pop('dying_file', DYING_FILE)
        self.timeline_config = kwargs.pop('timeline_config', {})
        warn_not_empty(kwargs)


class BasicDetector:
    def __init__(self, config: BasicDetectorConfig):
        """
        Detects novel words based on an earliness cutoff and separates dying
        and surviving words based on a lateness cutoff, based on a Timeline.

        Non-novel words are filtered out from the set of all word usages.

        :param config: see BasicDetectorConfig for details
        """
        self.config = config
        self.timeline = Timeline(TimelineConfig(**self.config.timeline_config))

    def run(self) -> None:
        usage_file = makepath(self.config.input_dir, self.config.usage_file)
        with open(usage_file, 'rb') as file:
            usage_dict = pickle.load(file)
        neologisms = dict((word, usage) for word, usage in usage_dict.items()
                          if not self.timeline.is_early(usage[0]))
        surviving, dying = {}, {}
        for word, usage in neologisms.items():
            if self.timeline.is_late(usage[1]):
                surviving[word] = usage
            else:
                dying[word] = usage
        self._save(surviving, self.config.surviving_file)
        self._save(dying, self.config.dying_file)

    def _save(self, words, filename):
        print(words)
        with open(makepath(self.config.output_dir, filename), 'wb') as file:
            pickle.dump(words, file, protocol=pickle.HIGHEST_PROTOCOL)
