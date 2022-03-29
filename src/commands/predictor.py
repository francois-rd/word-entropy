from commands.core import CommandBase
from predict.predict import Prediction, PredictionConfig


class PredictionCommand(CommandBase):
    @property
    def config_class(self):
        return PredictionConfig

    def start(self, config: PredictionConfig, parser_args):
        Prediction(parser_args.seed, config).run()
