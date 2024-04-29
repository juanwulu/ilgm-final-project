"""Dataset wrappers as Lightning datamodule."""

from __future__ import annotations, print_function

from collections.abc import Iterable
from multiprocessing import cpu_count
from typing import Optional

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from .dataset import INTERACTIONData, INTERACTIONDataset
from .subsampler import INTERACTIONSubSampler


class INTERACTIONDataModule(LightningDataModule):
    """A wrapper over `LightningDataModule` for INTERACTION dataset."""

    # ----------- public attributes ----------- #
    batch_size: int
    """int: batch size for data loading."""
    num_workers: int
    """int: number of workers for data loading."""
    pin_memory: bool
    """bool: if pin memory for data loading."""
    train_dataset: Optional[INTERACTIONDataset]
    """Optional[INTERACTIONDataset]: training datasets."""
    val_dataset: Optional[INTERACTIONDataset]
    """Optional[INTERACTIONDataset]: validation dataset."""
    test_dataset: Optional[INTERACTIONDataset]
    """Optional[INTERACTIONDataset]: test dataset."""

    def __init__(
        self,
        root: str,
        challenge_type: str,
        subsampler: INTERACTIONSubSampler = INTERACTIONSubSampler(),
        radius: Optional[float] = None,
        transform: Optional[BaseTransform | Iterable[BaseTransform]] = None,
        enable_train: bool = True,
        enable_val: bool = True,
        enable_test: bool = False,
        force_data_cache: bool = False,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Constructor function.

        Args:
            root (str): root directory of the INTERACTION dataset.
            challenge_type (str): name of the challenge, either `single-agent`,
                `conditional-single-agent`, `multi-agent`, or
                `conditional-multi-agent`.
            subsampler (INTERACTIONSubSampler, optional): subsampler for the
                dataset. Defaults to `INTERACTIONSubSampler()`.
            radius (Optional[float], optional): query range in meters. If it is
                `None`, keep all the available features. Defaults to `None`.
            transform (Optional[BaseTransform | Iterable[BaseTransform]],
                optional): transform for the dataset. Defaults to `None`.
            enable_meta_train (bool, optional): If further split the training
                dataset to support meta-training. Defaults to `False`.
            enable_train (bool, optional): If use training data.
                Defaults to `True`.
            enable_val (bool, optional): If use validation data.
                Defaults to `True`.
            enable_test (bool, optional): If use test data.
                Defaults to `False`.
            force_data_cache (bool, optional): if save cache tensors to local.
                Defaults to `False`.
            num_workers (Optional[int], optional): number of workers for data
                loading. If it is `None`, use the number of CPU cores.
                Defaults to `None`.
            pin_memory (bool, optional): if pin memory for data loading.
                Defaults to `True`.
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers or cpu_count()
        self.pin_memory = pin_memory

        if enable_train:
            self.train_dataset = INTERACTIONDataset(
                root=root,
                challenge_type=challenge_type,
                split="train",
                subsampler=subsampler,
                radius=radius,
                transform=transform,
                force_data_cache=force_data_cache,
            )
        else:
            self.train_dataset = None

        if enable_val:
            self.val_dataset = INTERACTIONDataset(
                root=root,
                challenge_type=challenge_type,
                split="val",
                subsampler=subsampler,
                radius=radius,
                transform=transform,
                force_data_cache=force_data_cache,
            )
        else:
            self.val_dataset = None

        if enable_test:
            self.test_dataset = INTERACTIONDataset(
                root=root,
                challenge_type=challenge_type,
                split="test",
                subsampler=subsampler,
                radius=radius,
                transform=transform,
                force_data_cache=force_data_cache,
            )
        else:
            self.test_dataset = None

    @property
    def num_train_data(self) -> int:
        return len(self.train_dataset)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset is None:
            raise ValueError("No training dataset!")
        shuffle = not isinstance(self.train_dataset, IterableDataset)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            follow_batch=INTERACTIONData.follow_batch,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            follow_batch=INTERACTIONData.follow_batch,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            follow_batch=INTERACTIONData.follow_batch,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
