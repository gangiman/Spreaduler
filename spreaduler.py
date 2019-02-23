import os
from typing import Any, Optional

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import socket
from datetime import datetime
from time import sleep
import traceback

_current_experiment = None  # type: Optional[ExperimentParams]


def _experiment_running():
    return _current_experiment is not None


def log_experiment_id(experiment_id: str):
    """
    Sets experiment id
    :param experiment_id:
    :return:
    """
    if _experiment_running():
        _current_experiment.log_experiment_id(experiment_id)


def log_status(status: str):
    """
    Sets status of the current experiment
    :param status:
    """
    if _experiment_running():
        _current_experiment.log_status(status)


def log_comment(text: str):
    """
    Sets comment of the current experiment
    """
    if _experiment_running():
        _current_experiment.log_comment(text)


def log_progress(current: int, max_value: int):
    """
    Sets progress of the current experiment
    """
    if _experiment_running():
        _current_experiment.log_progress(current, max_value)


def log_metric(metric: str, value: Any):
    """
    Logs metric of the current experiment
    :param metric: metric name
    :param value: metric value
    """
    if _experiment_running():
        _current_experiment.log_metric(metric, value)


class ParamsException(Exception):
    pass


class ParamsSheet(object):
    client_credentials = None
    params_sheet_id = None

    def __init__(self, parser, writable_column_types=None, experiment_id_column='experiment_id', server_name=None):
        """
        ParamsSheet object represents state of google spreadsheet with parameters and all methods to access it.
        :param parser: Argparse parser object with no positional arguments
        :param writable_column_types: dict of names of model metric columns and their types
        :param experiment_id_column: str name of experiment id column
        """
        assert isinstance(writable_column_types, dict), "writable_column_types has to be a dict"
        self.parser = parser
        self.experiment_id_column = experiment_id_column
        self.writable_column_types = writable_column_types
        self.column_types = writable_column_types.copy()
        self.update_type_from_parser()
        self.defaults = self.get_defaults_from_parser()
        if server_name is None:
            self.server = os.environ.get('SERVERNAME', None)
            if self.server is None:
                self.server = socket.gethostname()
        else:
            self.server = server_name + "_" + socket.gethostname()
        self._generate_credentials()

    def _generate_credentials(self):
        if isinstance(self.client_credentials, dict):
            credentials = ServiceAccountCredentials._from_parsed_json_keyfile(self.client_credentials,
                                                                              ['https://spreadsheets.google.com/feeds'])
        else:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(self.client_credentials,
                                                                           ['https://spreadsheets.google.com/feeds'])
        gc = gspread.authorize(credentials)
        self.sheet = gc.open_by_key(self.params_sheet_id).sheet1
        first_cell = self.sheet.cell(1, 1).value
        try:
            self.column_row_id = int(first_cell)
        except ValueError:
            self.column_row_id = 1
        self.columns = self.sheet.row_values(self.column_row_id)

    def get_params_for_training(self):
        full_params = self.get_table()
        params_to_process = full_params[
            (full_params.status == "") & (full_params.server == "")]
        params_for_this_server = full_params[
            (full_params.status == "") & (full_params.server == self.server)]
        if len(params_for_this_server):
            params_to_process = params_for_this_server
        if len(params_to_process) == 0:
            raise ParamsException("No params to process!")
        return ExperimentParams(
            params_to_process.index[0], dict(params_to_process.iloc[0]), self)

    def get_table(self):
        recs = self.sheet.get_all_records(head=self.column_row_id, default_blank="")
        return pd.DataFrame(recs)

    def update_cell(self, row, col, value):
        try:
            self.sheet.update_cell(row, col, value)
        except gspread.exceptions.APIError:
            self._generate_credentials()
            self.sheet.update_cell(row, col, value)

    def update_type_from_parser(self):
        self.column_types.update({
            _action.dest: _action.type
            for _action in self.parser._actions[1:]
        })
        # assert all(v is not None for v in self.column_types.values())

    def get_defaults_from_parser(self):
        return {_action.dest: _action.default
                for _action in self.parser._actions[1:]}

    def exec_loop(self, train_loop):
        """
        :param train_loop: train function that takes args namespace
        :return: None
        """
        global _current_experiment
        print("Starting worker...")
        timeout_power = 1
        while True:
            try:
                exp_params = self.get_params_for_training()
                _current_experiment = exp_params
                timeout_power = 4
                exp_params.log_status('running')
                exp_params.log_server()
                exp_params.log_time_to_column('time_started')
                train_loop(exp_params.args)
                exp_params.log_status('finished')
            except ParamsException:
                print("No params to process, waiting for {} seconds and checking again".format(2 ** timeout_power))
                sleep(2 ** timeout_power)
                if timeout_power <= 10:
                    timeout_power += 1
            except KeyboardInterrupt:
                exp_params.log_status('stopped')
                _current_experiment = None
            except Exception as e:
                tb = traceback.format_exc()
                exp_params.log_comment(tb)
                exp_params.log_status('error')
                _current_experiment = None


class ExperimentParams(object):
    _writable_columns = ('time_started', 'last_update', 'progress_bar', 'server', 'status', 'comment')

    def __init__(self, row_id, params, _params_sheet: ParamsSheet, fill_empty_with_defaults=True):
        self._params_sheet = _params_sheet
        self._params_row_id = row_id + 1 + _params_sheet.column_row_id
        self._read_only_columns = set(_params_sheet.columns) - set(self._writable_columns)
        default_args = _params_sheet.parser.parse_args()
        self.args = default_args
        for _idx, _wc_name in enumerate(_params_sheet.columns):
            if _wc_name in self._read_only_columns:
                if isinstance(params[_wc_name], str) and not params[_wc_name]:
                    if fill_empty_with_defaults:
                        self._params_sheet.update_cell(
                            self._params_row_id, _idx + 1, getattr(default_args, _wc_name, ''))
                else:
                    if _wc_name not in _params_sheet.column_types:
                        print("Unexpected column %s (known column: %s); casting it to str" %
                              (_wc_name, ",".join(_params_sheet.column_types.keys())))
                        setattr(self.args, _wc_name, str(params[_wc_name]))
                    else:
                        dtype = _params_sheet.column_types[_wc_name]
                        if dtype is None:  # type for bool is None
                            dtype = lambda x: x.lower() == 'true'
                        setattr(self.args, _wc_name, dtype(params[_wc_name]))

    def log_server(self):
        self._write_to_field('server', self._params_sheet.server)

    def log_time_to_column(self, column_name):
        now = datetime.now()
        self._write_to_field(column_name, now.strftime('%Y-%m-%d %H:%M:%S'))

    def log_status(self, status):
        self._write_to_field('status', status)

    def log_comment(self, text):
        self._write_to_field('comment', text)

    def log_progress(self, i, max_i):
        assert isinstance(i, int) and isinstance(max_i, int)
        if i == max_i:
            color = "green"
        else:
            color = "blue"
        formula = '=SPARKLINE({},{{"charttype","bar";"color1", "{}";"max",{}}})'.format(i, color, max_i)
        self._write_to_field('progress_bar', formula)
        self.log_time_to_column('last_update')

    def log_experiment_id(self, value):
        self._write_to_field(self._params_sheet.experiment_id_column, value)

    def log_metric(self, metric, value):
        if metric in self._params_sheet.writable_column_types:
            self._write_to_field(metric, value)
        self.log_time_to_column('last_update')

    def _write_to_field(self, column_name, value):
        row = self._params_row_id
        col = self._params_sheet.columns.index(column_name) + 1
        self._params_sheet.update_cell(row, col, value)
