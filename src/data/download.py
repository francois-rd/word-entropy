from pmaw import PushshiftAPI
import pandas as pd

from utils.pathing import makepath, DATA_DIR
from utils.misc import warn_not_empty


class RedditDownloaderConfig:
    def __init__(self, **kwargs):
        """
        Configs for the RedditDownloader class. Accepted kwargs are:

        output_dir: (type: Path-like, default: utils.pathing.DATA_DIR)
            Root directory in which to store all the output files.

        num_workers: (type: int, default: 1)
            The number of current threads to use for download.

        subreddits: (type: list, default: None)
            A string list of subreddits to download.

        after: (type: float, default: None)
            Restrict downloads to comments after this date (Epoch format).

        before: (type: float, default: None)
            Restrict downloads to comments before this date (Epoch format).

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        # NOTE: this assumes full path to files, not just filenames.
        self.output_dir = kwargs.pop('output_dir', str(DATA_DIR))
        self.num_workers = kwargs.pop('num_workers', 1)
        self.subreddits = kwargs.pop('subreddits', None)
        self.after = kwargs.pop('after', None)
        self.before = kwargs.pop('before', None)
        warn_not_empty(kwargs)


class RedditDownloader:
    def __init__(self, config: RedditDownloaderConfig):
        """
        Downloads (the subset of) Reddit from PushShift.io specified by the
        given configs.

        :param config: see RedditDownloaderConfig for details
        """
        self.config = config
        self.api = PushshiftAPI(num_workers=config.num_workers, jitter='full')

    def run(self) -> None:
        kwargs = {'filter_fn': self._filter_deleted_removed}
        if self.config.after is not None:
            kwargs['after'] = self.config.after
        if self.config.before is not None:
            kwargs['before'] = self.config.before
        if self.config.subreddits is None:
            comments = self.api.search_comments(**kwargs)
            self._save_comments([self._prune_fields(c) for c in comments])
        else:
            for sub in self.config.subreddits:
                comments = self.api.search_comments(subreddit=sub, **kwargs)
                comments = [self._prune_fields(c) for c in comments]
                self._save_comments(comments, comments[0]['subreddit_id'])

    def _save_comments(self, comments, subreddit=None):
        filename = "after{}before{}subreddit{}.csv".format(
            self.config.after, self.config.before, subreddit)
        filepath = makepath(self.config.output_dir, filename)
        df = pd.DataFrame(comments)
        df.to_csv(filepath, index=False, columns=list(df.axes[1]))

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
            'body': comment['body'],
            'subreddit_id': comment['subreddit_id'],
            'created_utc': comment['created_utc']
        }
