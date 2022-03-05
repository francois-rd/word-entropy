from commands.core import CommandBase
from model.distributions import Distributions, DistributionsConfig


class DistributionsCommand(CommandBase):
    @property
    def config_class(self):
        return DistributionsConfig

    def start(self, config: DistributionsConfig, parser_args):
        Distributions(config).run()
