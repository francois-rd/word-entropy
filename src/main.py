import argparse

from commands import RedditDownloaderCommand

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(
        description="Main entry point for running the experiments.")
    subparsers = main_parser.add_subparsers(title="main subcommands")

    # Preprocessing commands.
    RedditDownloaderCommand(subparsers.add_parser(
        'download', help="download a portion of the Reddit dataset"))

    args = main_parser.parse_args()
    args.func(args)
