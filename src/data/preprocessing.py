import pandas as pd
import spacy
import os
import re

from utils.pathing import makepath, RAW_DATA_DIR, PREPROC_DATA_DIR
from utils.misc import warn_not_empty


class RedditPreprocessorConfig:
    def __init__(self, **kwargs):
        """
        Configs for the RedditPreprocessor class. Accepted kwargs are:

        input_dir: (type: Path-like, default: utils.pathing.RAW_DATA_DIR)
            Root directory from which to read all the downloaded Reddit data.

        output_dir: (type: Path-like, default: utils.pathing.PREPROC_DATA_DIR)
            Root directory in which to store all the output files.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        # NOTE: this assumes full path to files, not just filenames.
        self.input_dir = kwargs.pop('input_dir', str(RAW_DATA_DIR))
        self.output_dir = kwargs.pop('output_dir', str(PREPROC_DATA_DIR))
        warn_not_empty(kwargs)


class RedditPreprocessor:
    def __init__(self, config: RedditPreprocessorConfig):
        """
        Preprocesses the (body of text from the) Reddit data using Spacy.

        :param config: see RedditPreprocessorConfig for details
        """
        self.config = config
        exclude = ['tok2vec', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
        self.nlp = spacy.load("en_core_web_sm", exclude=exclude)

    def run(self) -> None:
        for root, _, files in os.walk(self.config.input_dir):
            for file in files:
                df = pd.read_csv(makepath(root, file))
                df['body'] = df['body'].map(self._clean)
                path = makepath(self.config.output_dir, "cleaned-" + file)
                df.dropna().to_csv(path, index=False, columns=list(df.axes[1]))

    def _clean(self, body):
        kept = []
        for token in self.nlp(body):
            if token.is_alpha and not token.is_stop:
                # Collapse repeating letters to a maximum of 3.
                kept.append(re.sub(r'(.)\1\1+', r'\1\1\1', token.lower_))
        return " ".join(kept)


