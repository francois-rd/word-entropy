import argparse

from commands import RedditDownloaderCommand, RedditPreprocessorCommand

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(
        description="Main entry point for all programs.")
    subparsers = main_parser.add_subparsers(title="main subcommands")

    # Data gathering and cleaning commands.
    RedditDownloaderCommand(subparsers.add_parser(
        'download', help="download a portion of Reddit"))
    RedditPreprocessorCommand(subparsers.add_parser(
        'preprocess', help="preprocess the downloaded Reddit data"))

    args = main_parser.parse_args()
    args.func(args)
