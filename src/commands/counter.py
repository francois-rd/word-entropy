from commands.core import CommandBase
from data.count import RedditCounter, RedditCounterConfig


class RedditCounterCommand(CommandBase):
    @property
    def config_class(self):
        return RedditCounterConfig

    def start(self, config: RedditCounterConfig, parser_args):
        RedditCounter(config).run()
