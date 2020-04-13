from spreaduler import log_metric
from spreaduler import log_progress
from spreaduler import ParamsSheet
from random import randint
import pandas as pd
import argparse
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import pytest
import os

CREDENTIALS_FILE = os.environ.get('CREDENTIALS_FILE')
TEST_SHEET_ID = os.environ.get('TEST_SHEET_ID')
TEST_COLUMNS = ('some_param', 'accuracy', 'experiment_id', 'time_started',
                'last_update', 'progress_bar', 'server', 'status', 'comment')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--some_param", default=0, type=int)
    return parser


def train(args: argparse.Namespace):
    for i in range(2):
        log_progress(i + 1, 2)
        time.sleep(1)
    log_metric("accuracy", args.some_param / 10)


@pytest.fixture(scope="session")
def worksheet():
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        CREDENTIALS_FILE, ['https://spreadsheets.google.com/feeds']
    )
    gc = gspread.authorize(credentials)
    worksheet = gc.open_by_key(TEST_SHEET_ID).sheet1
    worksheet.clear()
    # Select a range
    cell_list = worksheet.range(1, 1, 1, 1 + len(TEST_COLUMNS))
    for cell, col in zip(cell_list, TEST_COLUMNS):
        cell.value = col
    # create test input params
    params_list = worksheet.range(2, 1, 3, 1)
    for _param_cell in params_list:
        _param_cell.value = randint(4, 100)
    # Update in batch
    worksheet.update_cells(cell_list + params_list)
    return worksheet


@pytest.fixture
def params(worksheet):
    return ParamsSheet(get_parser(), CREDENTIALS_FILE, TEST_SHEET_ID,
                       writable_column_types={'accuracy': float}, from_existing_worksheet=worksheet)


def test_params_sheet(params):
    assert tuple(params.columns) == TEST_COLUMNS


def test_train_loop(params):
    params.exec_loop(train, exit_when_no_params=True)
    full_spreadsheet = params.get_full_spreadsheet()
    pd.testing.assert_series_equal(full_spreadsheet.some_param / 10, full_spreadsheet.accuracy, check_names=False)
    assert (full_spreadsheet.status == 'finished').all()
