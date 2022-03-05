from redditcleaner import clean
from pmaw import PushshiftAPI
import pandas as pd
import re

from utils.pathing import makepath, RAW_DATA_DIR
from utils.timeline import TimelineConfig
from utils.misc import warn_not_empty


class RedditDownloaderConfig:
    def __init__(self, **kwargs):
        """
        Configs for the RedditDownloader class. Accepted kwargs are:

        output_dir: (type: Path-like, default: utils.pathing.RAW_DATA_DIR)
            Root directory in which to store all the output files.

        num_workers: (type: int, default: 1)
            The number of current threads to use for download.

        subreddits: (type: list, default: None)
            A string list of subreddits to download.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        # NOTE: this assumes full path to files, not just filenames.
        self.output_dir = kwargs.pop('output_dir', str(RAW_DATA_DIR))
        self.num_workers = kwargs.pop('num_workers', 1)
        self.subreddits = kwargs.pop('subreddits', None)
        self.timeline_config = kwargs.pop('timeline_config', {})
        warn_not_empty(kwargs)


class RedditDownloader:
    def __init__(self, config: RedditDownloaderConfig):
        """
        Downloads (the subset of) Reddit from PushShift.io specified by the
        given configs.

        :param config: see RedditDownloaderConfig for details
        """
        self.config = config
        self.timeline_config = TimelineConfig(**self.config.timeline_config)
        self.api = PushshiftAPI(num_workers=config.num_workers, jitter='full')

    def run(self) -> None:
        kwargs = {
            'after': self.timeline_config.start,
            'before': self.timeline_config.end,
            'filter_fn': self._filter_deleted_removed
        }
        if self.config.subreddits is None:
            comments = self.api.search_comments(**kwargs)
            self._save_comments([self._prune_fields(c) for c in comments])
        else:
            for sub in self.config.subreddits:
                comments = self.api.search_comments(subreddit=sub, **kwargs)
                comments = [self._prune_fields(c) for c in comments]
                self._save_comments(comments, comments[0]['subreddit_id'])

    def _save_comments(self, comments, subreddit=None):
        filename = "start{}end{}subreddit{}.csv".format(
            self.timeline_config.start, self.timeline_config.end, subreddit)
        filepath = makepath(self.config.output_dir, filename)
        df = pd.DataFrame(comments)
        df.dropna().to_csv(filepath, index=False, columns=list(df.axes[1]))

    @staticmethod
    def _filter_deleted_removed(comment):
        if comment['author'] in ['[removed]', '[deleted]']:
            return False
        if comment['body'] in ['[removed]', '[deleted]']:
            return False
        return True

    @staticmethod
    def _prune_fields(comment):
        return {
            'author_fullname': comment['author_fullname'],
            'body': RedditDownloader._clean_body(comment['body']),
            'subreddit_id': comment['subreddit_id'],
            'created_utc': comment['created_utc']
        }

    @staticmethod
    def _clean_body(body):
        # Quote has a bug. Matches "gt" within a word, not just "&gt;". The
        # simple fix employed here (not sure how robust it really is) is to
        # ensure there isn't a word character before it.
        cleanish = clean(body, quote=False)
        return re.sub(r'\W\"?\\?&?gt;?', '', cleanish)
