from spreaduler import log_metric, log_progress, ParamsSheet
import argparse
import time


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--some_param", default=0, type=int)
    return parser


def train(args: argparse.Namespace):
    for i in range(5):
        log_progress(i + 1, 5)
        time.sleep(0.2)
    log_metric("accuracy", args.some_param / 10)


if __name__ == '__main__':
    params = ParamsSheet(
        get_parser(),
        "Sample project-eb8164034438.json",
        '1xZHwmWTDWD3KBf9MpynCFYPTcsuJUEruhVDlQYTvbdo'
    )
    params.exec_loop(train)
