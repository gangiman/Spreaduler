import os

from spreaduler import ParamsSheet, log_metric, log_progress
import argparse
import time


class YourParamsSheet(ParamsSheet):
    params_sheet_id = '1xZHwmWTDWD3KBf9MpynCFYPTcsuJUEruhVDlQYTvbdo'
    client_credentials = "Sample project-eb8164034438.json"

    def __init__(self, parser):
        writable_metrics_and_types = {
            'accuracy': float
        }
        super(YourParamsSheet, self).__init__(
            parser,
            writable_column_types=writable_metrics_and_types,
            experiment_id_column='experiment_id')


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
    params = YourParamsSheet(get_parser())
    params.exec_loop(train)
