import pickle
from argparse import Namespace

import torch
import pytest
from pytorch_lightning import Trainer

from gspread_logger import PyDriveLogger
from tests.model_and_data_for_test import get_default_hparams
from tests.model_and_data_for_test import TestModelBase as LightningTestModel
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

#
# def test_pydrive():
#     gauth = GoogleAuth(settings_file='creds/pydrive_settings.yaml')
#     gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
#     drive = GoogleDrive(gauth)
#
#     # file1 = drive.CreateFile({'title': 'Hello.txt'})  # Create GoogleDriveFile instance with title 'Hello.txt'.
#     # file1.SetContentString('Hello World!') # Set content of the file from given string.
#     # file1.Upload()
#
#     file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
#     for file1 in file_list:
#         print('title: %s, id: %s' % (file1['title'], file1['id']))
#

def test_gspread_logger():
    """Verify that basic functionality of gspread logger works."""

    hparams = get_default_hparams()
    model = LightningTestModel(hparams)

    logger = PyDriveLogger(name='TestGDriveLogger',
                           settings_file='creds/pydrive_settings.yaml',
                           gspread_credentials='creds/gsheets_credentials.json')

    trainer_options = dict(max_epochs=1, train_percent_check=0.01, logger=logger, default_save_path='/tmp/')

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print("result finished")
    assert result == 1, "Training failed"

#
# def test_gspread_pickle(tmpdir):
#     """Verify that pickling trainer with gspread logger works."""
#
#     logger = GSpreadLogger(save_dir=tmpdir, name="gspread_pickle_test")
#
#     trainer_options = dict(max_epochs=1, logger=logger)
#
#     trainer = Trainer(**trainer_options)
#     pkl_bytes = pickle.dumps(trainer)
#     trainer2 = pickle.loads(pkl_bytes)
#     trainer2.logger.log_metrics({"acc": 1.0})
#
#
# def test_gspread_automatic_versioning(tmpdir):
#     """Verify that automatic versioning works"""
#
#     root_dir = tmpdir.mkdir("tb_versioning")
#     root_dir.mkdir("version_0")
#     root_dir.mkdir("version_1")
#
#     logger = GSpreadLogger(save_dir=tmpdir, name="tb_versioning")
#
#     assert logger.version == 2
#
#
# def test_gspread_manual_versioning(tmpdir):
#     """Verify that manual versioning works"""
#
#     root_dir = tmpdir.mkdir("tb_versioning")
#     root_dir.mkdir("version_0")
#     root_dir.mkdir("version_1")
#     root_dir.mkdir("version_2")
#
#     logger = GSpreadLogger(save_dir=tmpdir, name="tb_versioning", version=1)
#
#     assert logger.version == 1
#
#
# def test_gspread_named_version(tmpdir):
#     """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402' """
#
#     tmpdir.mkdir("tb_versioning")
#     expected_version = "2020-02-05-162402"
#
#     logger = GSpreadLogger(save_dir=tmpdir, name="tb_versioning", version=expected_version)
#     logger.log_hyperparams({"a": 1, "b": 2})  # Force data to be written
#
#     assert logger.version == expected_version
#     # Could also test existence of the directory but this fails
#     # in the "minimum requirements" test setup
#
#
# def test_gspread_no_name(tmpdir):
#     """Verify that None or empty name works"""
#
#     logger = GSpreadLogger(save_dir=tmpdir, name="")
#     assert logger.root_dir == tmpdir
#
#     logger = GSpreadLogger(save_dir=tmpdir, name=None)
#     assert logger.root_dir == tmpdir
#
#
# @pytest.mark.parametrize("step_idx", [10, None])
# def test_gspread_log_metrics(tmpdir, step_idx):
#     logger = GSpreadLogger(tmpdir)
#     metrics = {
#         "float": 0.3,
#         "int": 1,
#         "FloatTensor": torch.tensor(0.1),
#         "IntTensor": torch.tensor(1)
#     }
#     logger.log_metrics(metrics, step_idx)
#
#
# def test_gspread_log_hyperparams(tmpdir):
#     logger = GSpreadLogger(tmpdir)
#     hparams = {
#         "float": 0.3,
#         "int": 1,
#         "string": "abc",
#         "bool": True,
#         "dict": {'a': {'b': 'c'}},
#         "list": [1, 2, 3],
#         "namespace": Namespace(foo=Namespace(bar='buzz')),
#         "layer": torch.nn.BatchNorm1d
#     }
#     logger.log_hyperparams(hparams)
