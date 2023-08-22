import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import torch
from datasets import load_from_disk
from pytorch_lightning import LightningDataModule
from src.datamodules.fault_tolerant_sampler import (
    FaultTolerantDistributedSampler,
    RandomFaultTolerantSampler,
)
from src.utils.utils import get_logger
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer

logger = get_logger()


@dataclass
class SFTDataCollotor:
    max_length: int = 1024
    tokenizer: Callable = None

    def __init__(self, max_length, tokenizer, tensor_model_parallel_size):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.tensor_model_parallel_size = tensor_model_parallel_size

    def pad(self, ids: List, pad_token_id: int, max_length: int):
        if len(ids) > max_length:
            return ids[:max_length]
        return ids + [pad_token_id] * (max_length - len(ids))

    def __call__(self, samples):
        """use for dataloader to collate batch data

        Args:
            samples (list): list of samples
                a single sample is a dict with key 'data' and value is {"text": str, "mask": str}

        Returns:
            dict: {"input_ids": torch.tensor, "attention_mask": torch.tensor, "labels": torch.tensor}
        """
        batch_input_ids = []
        batch_label_ids = []
        for s in samples:
            input_ids = []
            label_ids = []
            for d in s["data"]:
                cur_input_ids = self.tokenizer.encode(d["text"], add_special_tokens=False)
                input_ids = input_ids + cur_input_ids
                if d["mask"] == "all":
                    label_ids = label_ids + [-100] * len(cur_input_ids)
                else:
                    mask_tokens = int(d["mask"])
                    label_ids = label_ids + [-100] * mask_tokens + cur_input_ids[mask_tokens:]
            batch_input_ids.append(input_ids)
            batch_label_ids.append(label_ids)
        # pad to max
        batch_max_length = (
            math.ceil(max([len(ids) for ids in batch_input_ids]) / self.tensor_model_parallel_size)
            * self.tensor_model_parallel_size
        )
        batch_max_length = min(batch_max_length, self.max_length)
        # input_ids left shift
        batch_input_ids = [
            self.pad(ids[:-1], self.tokenizer.pad_token_id, batch_max_length)
            for ids in batch_input_ids
        ]
        # label_ids right shift
        batch_label_ids = [self.pad(ids[1:], -100, batch_max_length) for ids in batch_label_ids]
        return torch.tensor(batch_input_ids), torch.tensor(batch_label_ids)


class SFTDataModule(LightningDataModule):
    # because the datasize in SFT is not to large, so we don't need to use shared memory
    def __init__(
        self,
        dataset_name,
        tokenizer_name,
        max_length=1024,
        cache_dir=None,
        batch_size=32,
        batch_size_eval=None,
        num_workers=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        fault_tolerant=False,
        ddp=False,
        fast_forward_epochs=None,
        fast_forward_batches=None,
        tensor_model_parallel_size=1,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        if tensor_model_parallel_size > 1:
            assert fault_tolerant and shuffle and ddp
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        self.collate_fn = SFTDataCollotor(
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            tensor_model_parallel_size=self.tensor_model_parallel_size,
        )

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        self.process_dataset()
        self.vocab_size = len(self.tokenizer)

    def process_dataset(self):
        if self.cache_dir is not None:  # load datasets(huggingface cache) from local path
            ds = load_from_disk(self.cache_dir)  # TODO: only train set now
            self.dataset_train, self.dataset_val, self.dataset_test = ds["train"], None, None
        else:
            raise NotImplementedError

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader"""
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            data_parallel_rank = None
            num_replicas = None
            if self.tensor_model_parallel_size > 1:
                from apex.transformer import parallel_state

                data_parallel_rank = parallel_state.get_data_parallel_rank()
                num_replicas = parallel_state.get_data_parallel_world_size()
            sampler = (
                FaultTolerantDistributedSampler(
                    self.dataset_train, rank=data_parallel_rank, num_replicas=num_replicas
                )
                if self.ddp
                else RandomFaultTolerantSampler(self.dataset_train)
            )
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if (
                self.ddp
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": self.fast_forward_batches * self.batch_size,
                    }
                )
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(
            self.dataset_train, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler
        )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader"""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader"""
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        sampler=None,
        collate_fn=None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data processed in fly
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"][
                "completed"
            ]
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint["loops"]["fit_loop"][
                "epoch_loop.batch_progress"
            ]["current"]["completed"]
        # At this point the train loader hasn't been constructed yet
