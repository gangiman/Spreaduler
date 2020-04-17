from argparse import Namespace
from typing import Optional, Dict, Union, Any, List, Tuple

import gspread
from collections import defaultdict
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import rowcol_to_a1
from gspread_formatting import set_column_width
from gspread_formatting import set_row_height

import logging
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

MIME_SPREADSHEET = 'application/vnd.google-apps.spreadsheet'
MIME_FOLDER = 'application/vnd.google-apps.folder'


class Table:
    """
    Abstraction for a table of metrics with the same schema
    """
    def __init__(self, ws: gspread.Worksheet,
                 header_row: int,
                 metrics: Union[Dict[str, Any], List[str]],
                 image_size: Optional[Tuple[int, int]] = None):
        """
        for_images - (height, width) in pixels of the images in column
        """
        self.cells_buffer = []
        self._worksheet = ws
        header = self._worksheet.row_values(header_row)
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

        if image_size is not None:
            height, width = image_size
            image_col = rowcol_to_a1(1, self._col_offset + int(self._metrics_header[0] == 'step'))
            set_row_height(self._worksheet, f'{header_row + 1}:{ws.row_count}', height)
            set_column_width(self._worksheet, image_col[:-1], width)

    def update(self, row: Dict[str, Any]):
        """
        Add metrics to a new row in a corresponding row
        """
        cell_list = self._worksheet.range(
            self._last_step, self._col_offset,
            self._last_step, self._col_offset + len(self._metrics_header) - 1)
        for _metric_name, _cell in zip(self._metrics_header, cell_list):
            v = row[_metric_name]
            if not isinstance(v, (int, float, str, bool)):
                v = str(v)
            _cell.value = v
            self.cells_buffer.append(_cell)
        self._last_step += 1

    def save(self):
        if self.cells_buffer:
            self._worksheet.update_cells(self.cells_buffer, value_input_option='USER_ENTERED')


class GSpreadLogger:
    _header_row: int
    _schema_to_table: Dict[frozenset, Table]
    _experiment_sheet: Optional[gspread.Spreadsheet]
    _hparams_ws: Optional[gspread.Worksheet]
    _metrics_ws: Optional[gspread.Worksheet]
    _images_ws: Optional[gspread.Worksheet]
    _log_folder_id: Optional[str]
    _root_folder_id: Optional[str]
    _image_folder_id: Optional[str]

    def __init__(self, name: Optional[str] = "lightning_logs", settings_file: str = 'settings.yaml',
                 version: Optional[Union[int, str]] = None, gspread_credentials: str = "credentials.json"):
        self._name = name
        self._images_ws = None
        self._metrics_ws = None
        self._hparams_ws = None
        self._log_folder_id = None
        self._root_folder_id = None
        self._image_folder_id = None
        gauth = GoogleAuth(settings_file=settings_file)
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            gspread_credentials, ['https://spreadsheets.google.com/feeds'])
        self.gc = gspread.authorize(credentials)
        self._version = version

    def _init_experiment_folder(self, version_override: str = None):
        if version_override is not None:
            self._version = version_override
        spreadsheet_file = self._find_in_gdrive(
            'logs_and_metrics', parent_dir_id=self.log_folder_id, mime=MIME_SPREADSHEET)
        if spreadsheet_file:  # creating new spreadsheet file
            spreadsheet_file = spreadsheet_file[0]
        else:
            spreadsheet_file = self.drive.CreateFile(
                {'title': 'logs_and_metrics', 'parents': [{'id': self.log_folder_id}],
                 'mimeType': MIME_SPREADSHEET})
            spreadsheet_file.Upload()
            spreadsheet_file.InsertPermission({
                'type': 'user',
                'value': self.gc.auth.service_account_email,
                'role': 'writer'})
        self._experiment_sheet = self.gc.open_by_key(spreadsheet_file['id'])
        worksheets = {_ws.title: _ws for _ws in self._experiment_sheet.worksheets()}
        self._hparams_ws = worksheets.pop('hyperparams', None)
        self._images_ws = worksheets.pop('images', None)
        self._metrics_ws = worksheets.pop('metrics', worksheets.pop('Sheet1', None))
        if self._metrics_ws is None:
            self._metrics_ws = self._experiment_sheet.get_worksheet(0)
        self._metrics_ws.update_title("metrics")
        self._schema_to_table = {}
        header_row_num = self._metrics_ws.acell("B1").value
        if header_row_num:
            self._header_row = int(header_row_num)
        else:
            self._header_row = 3
            self._metrics_ws.update('A1:B1', [["header_row", self._header_row]], value_input_option='USER_ENTERED')

    @property
    def root_folder_id(self):
        if self._root_folder_id is None:
            self._root_folder_id = self._find_or_create_folder(self._name)
        return self._root_folder_id

    @property
    def log_folder_id(self):
        if isinstance(self._version, str):
            version_folder = self._version
        else:
            version_folder = f"version_{self.version}"
        if self._log_folder_id is None:
            self._log_folder_id = self._find_or_create_folder(version_folder, parent_dir_id=self.root_folder_id)
        return self._log_folder_id

    @property
    def image_folder_id(self):
        if self._image_folder_id is None:
            self._image_folder_id = self._find_or_create_folder('images', parent_dir_id=self.log_folder_id)
        return self._image_folder_id

    def _find_in_gdrive(self, name: str = None, parent_dir_id: str = 'root',
                        mime: str = None, name_contains: str = None):
        """
        reference: https://developers.google.com/drive/api/v2/search-files
        """
        assert not (bool(name) and bool(name_contains)), "cannot process exact and substring search at the same time"
        query = f"'{parent_dir_id}' in parents and trashed=false"
        if mime is not None:
            query += f" and mimeType='{mime}'"
        if name_contains is not None:
            query += f" and title contains '{name_contains}'"
        elif name is not None:
            query += f" and title='{name}'"
        return self.drive.ListFile({'q': query}).GetList()

    def _get_table_for_set_of_metrics(self,
                                      metrics: Union[Dict[str, Any], List[str]],
                                      worksheet: str = 'metrics',
                                      image_size=None):
        if worksheet == 'metrics':
            ws = self._metrics_ws
        elif worksheet == 'images':
            ws = self._images_ws
        else:
            raise AssertionError(f'Unknown {worksheet}.')
        fs = frozenset(metrics.keys())
        if fs not in self._schema_to_table:
            self._schema_to_table[fs] = Table(ws, self._header_row, metrics, image_size=image_size)
        return self._schema_to_table[fs]

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._metrics_ws is None:
            self._init_experiment_folder()
        full_metrics = {**metrics, 'step': step}
        table = self._get_table_for_set_of_metrics(full_metrics)
        table.update(full_metrics)

    def save(self) -> None:
        for _t in self._schema_to_table.values():
            _t.save()

    def save_with_aggregate(self) -> None:
        aggregated_cell_buffers = defaultdict(list)
        worksheets = {}
        for _t in self._schema_to_table.values():
            ws_title = _t._ws.title
            aggregated_cell_buffers[ws_title] += _t.cells_buffer
            if ws_title not in worksheets:
                worksheets[ws_title] = _t._ws
        for _ws_title, _buffers in aggregated_cell_buffers.items():
            if _buffers:
                worksheets[_ws_title].update_cells(_buffers, value_input_option='USER_ENTERED')
                for _t in self._schema_to_table.values():
                    _t.cells_buffer = []

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        if self._hparams_ws is None:
            self._hparams_ws = self._experiment_sheet.add_worksheet(
                title="hyperparams", rows=40, cols=4)
        if isinstance(params, Namespace):
            params = params.__dict__
        hparams = [('hyperparameter', 'value')]
        for _key, _val in params.items():
            if not isinstance(_val, (str, int, float)):
                _val = str(_val)
            hparams.append((_key, _val))
        self._hparams_ws.update(f"A1:{rowcol_to_a1(len(hparams), 2)}",
                                hparams, value_input_option='USER_ENTERED')

    def log_image(self, tag, image, step=None, extension='png', image_size=None):
        """
        Uploads an image to experiment folder and adds it to the `images` worksheet
        Doc: https://support.google.com/docs/answer/3093333?hl=en
        """
        if self._images_ws is None:
            self._images_ws = self._experiment_sheet.add_worksheet(
                title="images", rows=100, cols=50)
        if step is None:
            step_str = ''
        else:
            step_str = "_" + str(step)
        image_file = self.drive.CreateFile(
            {'title': f'{tag}{step_str}.{extension}', 'parents': [{'id': self.image_folder_id}]})
        # Read file and set it as a content of this instance.
        image_file.SetContentFile(image)
        image_file.Upload()
        image_file.InsertPermission({
            'type': 'anyone',
            'value': 'anyone',
            'role': 'reader'})
        table = self._get_table_for_set_of_metrics({tag: None, 'step': step}, worksheet='images', image_size=image_size)
        table.update({
            tag: f'=IMAGE("https://docs.google.com/uc?export=download&id={image_file["id"]}")',
            'step': step
        })
        return image_file

    def _find_or_create_folder(self, name: str, parent_dir_id: str = 'root') -> str:
        folder = self._find_in_gdrive(name, parent_dir_id=parent_dir_id)
        if folder:
            folder = folder[0]
        else:
            dir_params = {'title': name, 'mimeType': MIME_FOLDER}
            if parent_dir_id != 'root':
                dir_params['parents'] = [{'id': parent_dir_id}]
            folder = self.drive.CreateFile(dir_params)
            folder.Upload()
        return folder['id']

    @staticmethod
    def is_dir(_file):
        return _file['mimeType'] == MIME_FOLDER

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        file_list = self._find_in_gdrive(
            mime=MIME_FOLDER, name_contains="version_", parent_dir_id=self.root_folder_id)
        existing_versions = [int(d['title'].split("_")[1]) for d in file_list]
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1

    @property
    def name(self) -> str:
        return self._name
