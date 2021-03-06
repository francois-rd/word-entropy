import argparse

from commands import (
    ExistingWordSamplerCommand,
    RedditDownloaderCommand,
    RedditPreprocessorCommand,
    RedditCounterCommand,
    WordUsageFinderCommand,
    BasicDetectorCommand,
    DistributionsCommand,
    TimeSeriesCommand,
    PlotTimeSeriesCommand,
    PlotStatsCommand,
    PredictionCommand
)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(
        description="Main entry point for all programs.")
    subparsers = main_parser.add_subparsers(title="main subcommands")

    # Data gathering and cleaning commands.
    ExistingWordSamplerCommand(subparsers.add_parser(
        'sample-existing', help="randomly samples a number of existing words"))
    RedditDownloaderCommand(subparsers.add_parser(
        'download', help="download a portion of Reddit"))
    RedditPreprocessorCommand(subparsers.add_parser(
        'preprocess', help="preprocess the downloaded Reddit data"))

    # Word usage finding and new word detection commands.
    RedditCounterCommand(subparsers.add_parser(
        'count', help="count words by user and by subreddit"))
    WordUsageFinderCommand(subparsers.add_parser(
        'find', help='find all usages of each word in the Reddit data'))
    BasicDetectorCommand(subparsers.add_parser(
        'basic-detect', help='detect new words from simple time slice cutoffs'))

    # Modeling commands.
    DistributionsCommand(subparsers.add_parser(
        'dists', help='compute word frequency distributions for all new words'))
    TimeSeriesCommand(subparsers.add_parser(
        'time-series', help='compute entropy time series from distributions'))

    # Analysis commands.
    PlotTimeSeriesCommand(subparsers.add_parser(
        'plot-ts', help='plot the multiple resulting time series'))
    PlotStatsCommand(subparsers.add_parser(
        'plot-stats', help='compute and plot stats from the time series'))
    PredictionCommand(subparsers.add_parser(
        'predict', help='predict word type from the resulting time series'))
    args = main_parser.parse_args()
    args.func(args)
