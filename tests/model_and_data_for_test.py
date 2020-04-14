import os
import urllib.request
from argparse import Namespace
from collections import OrderedDict
from typing import Tuple, Optional, Sequence

import torch
import torch.nn as nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule

from tests import TEST_ROOT

#: local path to test datasets
PATH_DATASETS = os.path.join(TEST_ROOT, 'Datasets')


class MNIST(Dataset):
    """
    Customized `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.

    Part of the code was copied from
    https://github.com/pytorch/vision/blob/build/v0.5.0/torchvision/datasets/mnist.py

    Args:
        root: Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    Examples:
        >>> dataset = MNIST(download=True)
        >>> len(dataset)
        60000
        >>> torch.bincount(dataset.targets)
        tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
    """

    RESOURCES = (
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    )

    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    cache_folder_name = 'complete'

    def __init__(self, root: str = PATH_DATASETS, train: bool = True,
                 normalize: tuple = (0.5, 1.0), download: bool = False):
        super().__init__()
        self.root = root
        self.train = train  # training set or test set
        self.normalize = normalize

        self.prepare_data(download)

        if not self._check_exists(self.cached_folder_path):
            raise RuntimeError('Dataset not found.')

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = torch.load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].float().unsqueeze(0)
        target = int(self.targets[idx])

        if self.normalize is not None:
            img = normalize_tensor(img, mean=self.normalize[0], std=self.normalize[1])

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.root, 'MNIST', self.cache_folder_name)

    def _check_exists(self, data_folder: str) -> bool:
        existing = True
        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            existing = existing and os.path.isfile(os.path.join(data_folder, fname))
        return existing

    def prepare_data(self, download: bool):
        if download:
            self._download(self.cached_folder_path)

    def _download(self, data_folder: str) -> None:
        """Download the MNIST data if it doesn't exist in cached_folder_path already."""

        if self._check_exists(data_folder):
            return

        os.makedirs(data_folder, exist_ok=True)

        for url in self.RESOURCES:
            fpath = os.path.join(data_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)


def normalize_tensor(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.sub_(mean).div_(std)
    return tensor


class TestingMNIST(MNIST):
    """Constrain image dataset

    Args:
        root: Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        num_samples: number of examples per selected class/digit
        digits: list selected MNIST digits/classes

    Examples:
        >>> dataset = TestingMNIST(download=True)
        >>> len(dataset)
        300
        >>> sorted(set([d.item() for d in dataset.targets]))
        [0, 1, 2]
        >>> torch.bincount(dataset.targets)
        tensor([100, 100, 100])
    """

    def __init__(self, root: str = PATH_DATASETS, train: bool = True,
                 normalize: tuple = (0.5, 1.0), download: bool = False,
                 num_samples: int = 100, digits: Optional[Sequence] = (0, 1, 2)):

        # number of examples per class
        self.num_samples = num_samples
        # take just a subset of MNIST dataset
        self.digits = digits if digits else list(range(10))

        self.cache_folder_name = 'digits-' + '-'.join(str(d) for d in sorted(self.digits)) \
                                 + f'_nb-{self.num_samples}'

        super().__init__(
            root,
            train=train,
            normalize=normalize,
            download=download
        )

    @staticmethod
    def _prepare_subset(full_data: torch.Tensor, full_targets: torch.Tensor,
                        num_samples: int, digits: Sequence):
        classes = {d: 0 for d in digits}
        indexes = []
        for idx, target in enumerate(full_targets):
            label = target.item()
            if classes.get(label, float('inf')) >= num_samples:
                continue
            indexes.append(idx)
            classes[label] += 1
            if all(classes[k] >= num_samples for k in classes):
                break
        data = full_data[indexes]
        targets = full_targets[indexes]
        return data, targets

    def prepare_data(self, download: bool) -> None:
        if self._check_exists(self.cached_folder_path):
            return
        if download:
            self._download(super().cached_folder_path)

        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            data, targets = torch.load(os.path.join(super().cached_folder_path, fname))
            data, targets = self._prepare_subset(data, targets, self.num_samples, self.digits)
            torch.save((data, targets), os.path.join(self.cached_folder_path, fname))


class TestModelBase(LightningModule):
    """Base LightningModule for testing. Implements only the required interface."""

    def __init__(self, hparams, force_remove_distributed_sampler: bool = False):
        """Pass in parsed HyperOptArgumentParser to the model."""
        # init superclass
        super().__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand(5, 28 * 28)

        # remove to test warning for dist sampler
        self.force_remove_distributed_sampler = force_remove_distributed_sampler

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """Layout model."""
        self.c_d1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.out_features)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """No special modification required for lightning, define as you normally would."""
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)

        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)

        return logits

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Lightning calls this inside the training loop"""
        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        # alternate possible outputs to test
        if self.trainer.batch_idx % 1 == 0:
            output = OrderedDict({
                'loss': loss_val,
                'progress_bar': {'some_val': loss_val * loss_val},
                'log': {'train_some_val': loss_val * loss_val},
            })

            return output
        if self.trainer.batch_idx % 2 == 0:
            return loss_val

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        # try no scheduler for this model (testing purposes)
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        _ = TestingMNIST(root=self.hparams.data_root, train=True, download=True)

    def _dataloader(self, train):
        # init data generators
        dataset = TestingMNIST(root=self.hparams.data_root, train=train, download=False)
        # when using multi-node we need to add the datasampler
        batch_size = self.hparams.batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(True)


def _get_output_metric(output, name):
    if isinstance(output, dict):
        val = output[name]
    else:  # if it is 2level deep -> per dataloader and per batch
        val = sum(out[name] for out in output) / len(output)
    return val


def get_default_hparams(continue_training=False, hpc_exp_number=0):
    _ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    args = {
        'drop_prob': 0.2,
        'batch_size': 32,
        'in_features': 28 * 28,
        'learning_rate': 0.001 * 8,
        'optimizer_name': 'adam',
        'data_root': PATH_DATASETS,
        'out_features': 10,
        'hidden_dim': 1000,
    }

    if continue_training:
        args['test_tube_do_checkpoint_load'] = True
        args['hpc_exp_number'] = hpc_exp_number

    hparams = Namespace(**args)
    return hparams
