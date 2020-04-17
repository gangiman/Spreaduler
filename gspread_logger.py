from argparse import Namespace
from typing import Optional, Dict, Union, Any, List

import gspread
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import rowcol_to_a1

import logging
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)


class Table:
    """
    Abstraction for a table of metrics with the same schema
    """
    def __init__(self, ws: gspread.Worksheet, header_row: int, metrics: Union[Dict[str, Any], List[str]]):
        self.cells_buffer = []
        self._metrics_worksheet = ws
        header = self._metrics_worksheet.row_values(header_row)
        if not header:
            self._col_offset = 1
        else:
            self._col_offset = len(header) + 2
        self._last_step = header_row
        self._metrics_header = []
        _metrics = metrics.copy()
        if isinstance(_metrics, dict):
            step = _metrics.pop('step')
            _metrics = list(_metrics.keys())
        else:
            step = None
            index = _metrics.index('step')
            if index >= 0:
                del _metrics[index]
        if step is not None:
            self._metrics_header += ['step']
        self._metrics_header += _metrics
        self.update({_h: _h for _h in self._metrics_header})

    def update(self, row: Dict[str, Any]):
        """
        Add metrics to a new row in a corresponding row
        """
        cell_list = self._metrics_worksheet.range(
            self._last_step, self._col_offset,
            self._last_step, self._col_offset + len(self._metrics_header) - 1)
        for _metric_name, _cell in zip(self._metrics_header, cell_list):
            v = row[_metric_name]
            if not isinstance(v, (int, float, str, bool)):
                v = str(v)
            _cell.value = v
            self.cells_buffer.append(_cell)
        self._last_step += 1


class GSpreadLogger:
    _header_row: int
    _schema_to_table: Dict[frozenset, Table]
    _experiment_sheet: Optional[gspread.Spreadsheet]
    _hparams_worksheet: Optional[gspread.Worksheet]
    MIME_SPREADSHEET = 'application/vnd.google-apps.spreadsheet'

    def __init__(self, name: Optional[str] = "lightning_logs", settings_file: str = 'settings.yaml',
                 version: Optional[Union[int, str]] = None, gspread_credentials: str = "credentials.json"):
        self._name = name
        gauth = GoogleAuth(settings_file=settings_file)
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            gspread_credentials, ['https://spreadsheets.google.com/feeds'])
        self.gc = gspread.authorize(credentials)
        if version is None:
            self._version = self._get_next_version()
        else:
            self._version = version
        if isinstance(self._version, str):
            version_folder = self.version
        else:
            version_folder = f"version_{self._version}"
        self._version = version
        self.root_folder_id = self._find_or_create_folder(name)
        self.log_folder_id = self._find_or_create_folder(version_folder, parent_dir_id=self.root_folder_id)
        spreadsheet_file = self._find_in_gdrive('logs_and_metrics', parent_dir_id=self.log_folder_id,
                                                mime=self.MIME_SPREADSHEET)
        if spreadsheet_file is None:  # creating new spreadsheet file
            spreadsheet_file = self.drive.CreateFile(
                {'title': 'logs_and_metrics', 'parents': [{'id': self.log_folder_id}],
                 'mimeType': 'application/vnd.google-apps.spreadsheet'})
            spreadsheet_file.Upload()
            spreadsheet_file.InsertPermission({
                'type': 'user',
                'value': self.gc.auth.service_account_email,
                'role': 'writer'})
        self._experiment_sheet = self.gc.open_by_key(spreadsheet_file['id'])
        worksheets = {_ws.title: _ws for _ws in self._experiment_sheet.worksheets()}
        self._hparams_worksheet = worksheets.pop('hyperparams', None)
        metrics_worksheet = worksheets.pop('metrics', worksheets.pop('Sheet1', None))
        if self._hparams_worksheet is None:
            self._hparams_worksheet = self._experiment_sheet.add_worksheet(
                title="hyperparams", rows=40, cols=4)
        if metrics_worksheet is None:
            metrics_worksheet = self._experiment_sheet.get_worksheet(0)
        metrics_worksheet.update_title("metrics")

        self._ws = metrics_worksheet
        self._schema_to_table = {}
        self._header_row = 3
        hr = self._ws.acell("B1").value
        if hr:
            self._header_row = int(hr)
        self._ws.update('A1:B1', [["header_row", self._header_row]], value_input_option='USER_ENTERED')

    def _find_in_gdrive(self, name: str, parent_dir_id: str = 'root', mime='application/vnd.google-apps.folder'):
        output = None
        file_list = self.drive.ListFile(
            {'q': f"'{parent_dir_id}' in parents and trashed=false and title='{name}'"}).GetList()
        for _file in file_list:
            if _file['mimeType'] == mime:
                output = _file
                break
        return output

    def _create_table_from_keys(self, metrics):
        return Table(self._ws, self._header_row, metrics)

    def _get_table_for_set_of_metrics(self, metrics: Dict[str, Any]):
        fs = frozenset(metrics.keys())
        if fs not in self._schema_to_table:
            self._schema_to_table[fs] = self._create_table_from_keys(metrics)
        return self._schema_to_table[fs]

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        full_metrics = {**metrics, 'step': step}
        table_rc = self._get_table_for_set_of_metrics(full_metrics)
        table_rc.update(full_metrics)

    def save(self) -> None:
        aggregated_cell_buffers = []
        for _t in self._schema_to_table.values():
            aggregated_cell_buffers += _t.cells_buffer
        if aggregated_cell_buffers:
            self._ws.update_cells(aggregated_cell_buffers, value_input_option='USER_ENTERED')
            for _t in self._schema_to_table.values():
                _t.cells_buffer = []

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        if isinstance(params, Namespace):
            params = params.__dict__
        hparams = [('hyperparameter', 'value')]
        for _key, _val in params.items():
            if not isinstance(_val, (str, int, float)):
                _val = str(_val)
            hparams.append((_key, _val))
        self._hparams_worksheet.update(f"A1:{rowcol_to_a1(len(hparams), 2)}",
                                       hparams, value_input_option='USER_ENTERED')

    def _find_or_create_folder(self, name: str, parent_dir_id: str = 'root') -> str:
        folder = self._find_in_gdrive(name, parent_dir_id=parent_dir_id)
        if folder is None:
            dir_params = {'title': name, 'mimeType': 'application/vnd.google-apps.folder'}
            if parent_dir_id != 'root':
                dir_params['parents'] = [{'id': parent_dir_id}]
            folder = self.drive.CreateFile(dir_params)
            folder.Upload()
        return folder['id']

    @staticmethod
    def is_dir(_file):
        return _file['mimeType'] == 'application/vnd.google-apps.folder'

    @property
    def version(self) -> int:
        return self._version

    def _get_next_version(self):
        existing_versions = []
        file_list = self.drive.ListFile({'q': f"'{self.root_folder_id}' in parents and trashed=false"}).GetList()
        for d in file_list:
            title = d['title']
            if self.is_dir(d) and title.startswith("version_"):
                existing_versions.append(int(title.split("_")[1]))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1

    @property
    def name(self) -> str:
        return self._name
