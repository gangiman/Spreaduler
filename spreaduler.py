import os
from typing import Any, Optional, List, Dict

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
    def __init__(self, parser, client_credentials: str, params_sheet_id = None,
                 writable_column_types: Dict[str, Any]=None, experiment_id_column: str='experiment_id',
                 server_name: str=None, fill_empty_with_defaults=True):
        """
        ParamsSheet object represents state of google spreadsheet with parameters and all methods to access it.
        :param parser: Argparse parser object with no positional arguments
        :param writable_column_types: dict of names of model metric columns and their types.
        If it is None, writing unknown metrics won't generate warnings.
        :param experiment_id_column: str name of experiment id column.
        """
        self.client_credentials = client_credentials
        self.params_sheet_id = params_sheet_id
        self.experiment_id_column = experiment_id_column
        self.fill_empty_with_defaults = fill_empty_with_defaults
        self.parser = parser
        self.defaults = {
            _action.dest: _action.default
            for _action in self.parser._actions[1:]
        }

        # Setting up dictionaries holding column types
        self.metric_column_types = writable_column_types
        if self.metric_column_types is None:
            self.column_types = {}
        else:
            assert isinstance(self.metric_column_types, dict), "writable_column_types has to be a dict"
            self.column_types = writable_column_types.copy()
        self.column_types.update({
            _action.dest: _action.type
            for _action in self.parser._actions[1:]
        })

        # Getting server name
        if server_name is None:
            self.server = os.environ.get('SERVERNAME', None)
            if self.server is None:
                self.server = socket.gethostname()
        else:
            self.server = server_name + "_" + socket.gethostname()

        # Getting credentials
        self._generate_credentials()

    def update_cell(self, row, column_name, value):
        col = self.columns.index(column_name) + 1
        try:
            self.sheet.update_cell(row, col, value)
        except gspread.exceptions.APIError:
            self._generate_credentials()
            self.sheet.update_cell(row, col, value)

    def exec_loop(self, train_loop):
        """
        :param train_loop: train function that takes args namespace
        :return: None
        """
        global _current_experiment
        print("Starting worker...")
        timeout_power = 1
        while True:
            exp_params = None
            try:
                exp_params = self._get_params_for_training()
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
                if exp_params is not None:
                    exp_params.log_status('stopped')
                _current_experiment = None
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                if exp_params is not None:
                    exp_params.log_comment(tb)
                    exp_params.log_status('error')
                _current_experiment = None

    def _generate_credentials(self):
        if isinstance(self.client_credentials, dict):
            credentials = ServiceAccountCredentials._from_parsed_json_keyfile(
                self.client_credentials, ['https://spreadsheets.google.com/feeds']
            )
        else:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                self.client_credentials,['https://spreadsheets.google.com/feeds']
            )
        gc = gspread.authorize(credentials)
        self.sheet = gc.open_by_key(self.params_sheet_id).sheet1
        first_cell = self.sheet.cell(1, 1).value
        try:
            self.column_row_id = int(first_cell)
        except ValueError:
            self.column_row_id = 1
        self.columns = self.sheet.row_values(self.column_row_id)

    def _get_params_for_training(self) -> "ExperimentParams":
        """
        This function gets next experiment parameter settings
        """
        try:
            all_spreadsheet = pd.DataFrame(self.sheet.get_all_records(head=self.column_row_id, default_blank=""))
        except gspread.exceptions.APIError:
            self._generate_credentials()
            all_spreadsheet = pd.DataFrame(self.sheet.get_all_records(head=self.column_row_id, default_blank=""))
        params_to_process = all_spreadsheet[
            (all_spreadsheet.status == "") & (all_spreadsheet.server == "")]
        params_for_this_server = all_spreadsheet[
            (all_spreadsheet.status == "") & (all_spreadsheet.server == self.server)]
        if len(params_for_this_server):
            params_to_process = params_for_this_server
        if len(params_to_process) == 0:
            raise ParamsException("No params to process!")
        return ExperimentParams(
            params_to_process.index[0] + 1 + self.column_row_id,
            dict(params_to_process.iloc[0]),
            self, self.fill_empty_with_defaults)


class ExperimentParams(object):
    _system_columns = ('time_started', 'last_update', 'progress_bar', 'server', 'status', 'comment')

    def __init__(self, row_id, params, _params_sheet: ParamsSheet, fill_empty_with_defaults: bool):
        self._params_sheet = _params_sheet
        self._params_row_id = row_id
        self.args = _params_sheet.parser.parse_args()
        parameter_columns = set(_params_sheet.columns) - set(self._system_columns) - {_params_sheet.experiment_id_column}
        for _column_name in parameter_columns:
            # Skipping metric columns
            if _params_sheet.metric_column_types is not None and _column_name in _params_sheet.metric_column_types:
                continue

            parameter_value = params[_column_name]
            if isinstance(params[_column_name], str) and (params[_column_name] == ""):
                # This parameter field is empty
                if fill_empty_with_defaults:
                    parameter_value = getattr(self.args, _column_name, '')
                    self._params_sheet.update_cell(
                        self._params_row_id, _column_name,
                        parameter_value
                    )

            if _column_name not in _params_sheet.column_types:
                print("Unexpected column %s (known column: %s)" % (_column_name, ",".join(_params_sheet.column_types)))
                continue

            dtype = _params_sheet.column_types[_column_name]
            if dtype is None:  # type for bool is None
                dtype = lambda x: x.lower() == 'true'
            setattr(self.args, _column_name, dtype(parameter_value))

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
        if (self._params_sheet.metric_column_types is not None) and \
                (metric not in self._params_sheet.metric_column_types):
            print("Logging unknown metric %s" % metric)
        self._write_to_field(metric, value)
        self.log_time_to_column('last_update')

    def _write_to_field(self, column_name, value):
        self._params_sheet.update_cell(
            self._params_row_id,
            column_name,
            value)
