from os.path import dirname, realpath, isfile
import pickle
from pathlib import Path
from collections import defaultdict
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from datasets import load_dataset
from numpy import log
from torchtext.vocab import CharNGram

CACHE_DIR = Path(dirname(realpath(__file__))) / 'data'
DATA_DIR = CACHE_DIR / "conll_data"
VECTORS_DIR = CACHE_DIR / ".vector_cache"
WORD_COUNT_DATA = CACHE_DIR / 'word_count.pickle'

VECTORS = CharNGram

class HuggingfaceDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config['batch_size']
        self.num_workers = 88 # config['num_workers']

    def prepare_data(self):
        # download only
        self.download_and_preprocess()

    @staticmethod
    @abstractmethod
    def download_and_preprocess(ds):
        raise NotImplementedError()

    def setup(self, stage):
        ds = self.download_and_preprocess()

        # Splits
        split_sizes = [int(len(ds) * 0.7), int(len(ds) * 0.15), int(len(ds) * 0.15)]
        if sum(split_sizes) < len(ds):
            split_sizes[0] += len(ds) - sum(split_sizes)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(ds, split_sizes)

    def split_dataloader(self, split):
        return DataLoader(split, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        return self.split_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.split_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.split_dataloader(self.test_dataset)

class WordCountDataModule(HuggingfaceDataModule):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def download_and_preprocess(ds_name="conll2003"):
        if isfile(WORD_COUNT_DATA):
            with open(WORD_COUNT_DATA, 'rb') as ds_pickle:
                word_count_ds = pickle.load(ds_pickle)

        else:
            conll_dataset = load_dataset(ds_name)
            embedder = VECTORS(cache=VECTORS_DIR)

            counts = defaultdict(int)
            for split in 'train', 'test', 'validation':
                for example in conll_dataset[split]:
                    for token in example['tokens']:
                        counts[token.lower()] += 1
            word_count_ds = []
            for t, c in counts.items():
                word_count_ds.append((embedder[t].float(), torch.tensor([log(c)], dtype=torch.float)))

            with open(WORD_COUNT_DATA, 'wb') as ds_pickle:
                pickle.dump(word_count_ds, ds_pickle)
        return word_count_ds

