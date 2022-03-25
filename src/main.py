import argparse

from commands import (
    ExistingWordSamplerCommand,
    RedditDownloaderCommand,
    RedditPreprocessorCommand,
    WordUsageFinderCommand,
    BasicDetectorCommand,
    DistributionsCommand,
    TimeSeriesCommand
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
    WordUsageFinderCommand(subparsers.add_parser(
        'find', help='find all usages of each word in the Reddit data'))
    BasicDetectorCommand(subparsers.add_parser(
        'basic-detect', help='detect new words from simple time slice cutoffs'))

    # Modeling commands.
    DistributionsCommand(subparsers.add_parser(
        'dists', help='compute word frequency distributions for all new words'))
    TimeSeriesCommand(subparsers.add_parser(
        'time-series', help='compute entropy time series from distributions'))

    args = main_parser.parse_args()
    args.func(args)
