"""One installed entry point; task modules also retain their python -m API."""

import argparse
import importlib


COMMANDS = {
    "train": "multicamera4rc.train",
    "infer": "multicamera4rc.infer",
    "evaluate": "multicamera4rc.evaluate",
    "launch": "multicamera4rc.launch",
    "demo": "multicamera4rc.demo",
    "adapter-demo": "multicamera4rc.datasets.demo",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="MultiCamera4RC training, inference and validation")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="Pass --help after the command for its options")
    args = parser.parse_args(argv)
    result = importlib.import_module(COMMANDS[args.command]).main(args.arguments)
    # Existing Python APIs return artifact Paths. Console entry points must
    # return an exit status, not pass those Paths to sys.exit as errors.
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
