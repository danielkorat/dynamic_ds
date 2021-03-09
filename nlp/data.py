import os

import torch
from torch import tensor
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from numpy import log
from torchtext.vocab import CharNGram
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / 'data'

class WordCountDataModule(pl.LightningDataModule):
    def __init__(self, config, embedder_cls=CharNGram):
        super().__init__()
        self.batch_size = config['batch_size']
        self.embedder_cls = embedder_cls

    def add_unique_counts_and_embeds(self, ds):
        counts = defaultdict(int)
        embedder = self.embedder_cls(cache=DATA_DIR / "vector_cache")
        for split in 'train', 'test', 'validation':
            for example in tqdm(ds[split]):
                for token in example['tokens']:
                    counts[token.lower()] += 1
        ds = []
        for t, c in counts.items():
            ds.append((embedder[t].float(), tensor([log(c)], dtype=torch.float)))
        return ds

    def prepare_data(self):
        # download only
        self.download_data()

    @staticmethod
    def download_data():
        CharNGram(cache=DATA_DIR / "vector_cache")
        load_dataset("conll2003", data_dir=DATA_DIR / "conll_data")

    def setup(self, stage):
        conll_dataset = load_dataset("conll2003")
        ds = self.add_unique_counts_and_embeds(conll_dataset)
        split_sizes = [int(len(ds) * 0.7), int(len(ds) * 0.15), int(len(ds) * 0.15)]
        if sum(split_sizes) < len(ds):
            split_sizes[0] += len(ds) - sum(split_sizes)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(ds, split_sizes)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__=="__main__":
    DS_NAME = "conll2003"

    counts = defaultdict(int)
    ds = load_dataset(DS_NAME)
    for split in 'train', 'test', 'validation':
        for example in tqdm(ds[split]):
            for token in example['tokens']:
                counts[token.lower()] += 1

    log_counts = defaultdict(int)
    for w, count in counts.items():
        log_counts[w] = log(count)
    sorted_counts = sorted(log_counts.values(), reverse=True)
    df = pd.DataFrame(data=sorted_counts)
    df.index = log(df.index)

    df.plot()
    plt.xlabel("sorted items in log scale")
    plt.ylabel("frequency in log scale")
    plt.legend([DS_NAME])
    plt.show()
