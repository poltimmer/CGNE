import os
import random
from typing import List, Tuple, Optional

import wandb
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from CGNE.dataset import SequenceHDF5Dataset, padding_collate_fn


class CGNEDataModule(LightningDataModule):
    def __init__(self, data_dir: str, train_split: float = 0.6, val_split: float = 0.2, test_split: float = 0.2,
                 batch_size: int = 16, num_workers: int = 0,
                 sequence_length: int = 30, overlap: int = 1, stride: int = 1, y_scramble: Optional[str] = None):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_classes, self.val_classes, self.test_classes = self._get_class_split()
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.stride = stride
        self.y_scramble = y_scramble

    def _get_class_split(self) -> Tuple[List[str], List[str], List[str]]:
        # Split entire sequences for train/val/test
        all_classes = sorted(os.listdir(self.data_dir))
        random.shuffle(all_classes)
        train_count = int(len(all_classes) * self.train_split)
        val_count = int(len(all_classes) * self.val_split)
        test_count = min(int(len(all_classes) * self.test_split), len(all_classes) - train_count - val_count)

        train_classes = all_classes[:train_count]
        val_classes = all_classes[train_count:train_count + val_count]
        test_classes = all_classes[train_count + val_count:train_count + val_count + test_count]
        return train_classes, val_classes, test_classes

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            wandb.log({
                "train_classes": self.train_classes,
                "val_classes": self.val_classes,
                "test_classes": self.test_classes
            })

        self._set_datasets(SequenceHDF5Dataset, sequence_length=self.sequence_length, overlap=self.overlap,
                           stride=self.stride, use_step_number=False, y_scramble=self.y_scramble)

    def _set_datasets(self, dataset_class, **kwargs):
        self.train_dataset = dataset_class(self.data_dir, specific_classes=self.train_classes, **kwargs)
        self.val_dataset = dataset_class(self.data_dir, specific_classes=self.val_classes, **kwargs)
        self.test_dataset = dataset_class(self.data_dir, specific_classes=self.test_classes, **kwargs)

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          collate_fn=padding_collate_fn)
