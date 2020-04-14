from argparse import Namespace
from typing import Optional, Dict, Union, Any

import torch
import gspread
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pytorch_lightning.loggers import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from oauth2client.service_account import ServiceAccountCredentials

import logging
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)


class PyDriveLogger(LightningLoggerBase):
    _experiment_sheet: Optional[gspread.Spreadsheet]

    def __init__(self, name: Optional[str] = "lightning_logs", settings_file: str = 'settings.yaml',
                 version: Optional[Union[int, str]] = None, gspread_credentials: str = "credentials.json"):
        super().__init__()
        self._name = name
        gauth = GoogleAuth(settings_file=settings_file)
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)

        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            gspread_credentials, ['https://spreadsheets.google.com/feeds']
        )
        self.gc = gspread.authorize(credentials)
        self.spreadsheet_file = None
        self.root_folder_id = None
        self._log_folder_id = None
        self._experiment_sheet = None
        self._version = version
        file_list = self.drive.ListFile(
            {'q': f"'root' in parents and trashed=false and title='{name}'"}).GetList()
        for _file in file_list:
            if self.is_dir(_file):
                self.root_folder_id = _file['id']
                break
        if self.root_folder_id is None:
            root_folder = self.drive.CreateFile({'title': name, 'mimeType': 'application/vnd.google-apps.folder'})
            root_folder.Upload()
            self.root_folder_id = root_folder['id']

    @staticmethod
    def is_dir(_file):
        return _file['mimeType'] == 'application/vnd.google-apps.folder'

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
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
    def log_folder_id(self) -> str:
        if self._log_folder_id is None:
            version = self.version if isinstance(self.version, str) else f"version_{self.version}"
            log_dir = self.drive.CreateFile({
                'title': version,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [{'id': self.root_folder_id}]})
            log_dir.Upload()
            self._log_folder_id = log_dir['id']
        return self._log_folder_id

    def _create_spreadsheet(self):
        if self._experiment_sheet is None:
            spreadsheet_file = self.drive.CreateFile(
                {'title': 'logs_and_metrics', 'parents': [{'id': self.log_folder_id}],
                 'mimeType': 'application/vnd.google-apps.spreadsheet'})
            spreadsheet_file.Upload()
            spreadsheet_file.InsertPermission({
                'type': 'user',
                'value': self.gc.auth.service_account_email,
                'role': 'writer'})
            self._experiment_sheet = self.gc.open_by_key(spreadsheet_file['id'])

    @property
    def experiment(self):
        self._create_spreadsheet()
        return self

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        if isinstance(params, Namespace):
            params = params.__dict__
        n_params = len(params)
        self._create_spreadsheet()
        worksheet = self._experiment_sheet.add_worksheet(title="hyperparams", rows=max(20, n_params + 2), cols=4)
        cell_list = worksheet.range(1, 1, n_params, 2)
        for (_cell_key, _cell_val), (_key, _val) in zip(zip(*[iter(cell_list)] * 2), params.items()):
            _cell_key.value = _key
            if not isinstance(_val, (str, int, float)):
                _val = str(_val)
            _cell_val.value = _val
        worksheet.update_cells(cell_list)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            # TODO replace self.experiment.add_scalar(k, v, step)

    @rank_zero_only
    def save(self) -> None:
        # TODO fix
        pass

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def name(self) -> str:
        return self._name
