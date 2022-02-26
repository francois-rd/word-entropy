from commands.core import CommandBase
from data.preprocessing import RedditPreprocessor, RedditPreprocessorConfig


class RedditPreprocessorCommand(CommandBase):
    @property
    def config_class(self):
        return RedditPreprocessorConfig

    def start(self, config: RedditPreprocessorConfig, parser_args):
        RedditPreprocessor(config).run()
