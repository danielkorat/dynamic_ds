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
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

CACHE_DIR = Path(dirname(realpath(__file__))) / 'data'
VECTORS_DIR = CACHE_DIR / ".vector_cache"

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
    def download_and_preprocess():
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
    ds_name = "conll2003"
    ds_cache = CACHE_DIR / 'word_count.pickle'
    data_dir = CACHE_DIR / 'conll_data'

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def download_and_preprocess():
        if isfile(WordCountDataModule.ds_cache):
            with open(WordCountDataModule.ds_cache, 'rb') as ds_pickle:
                word_count_ds = pickle.load(ds_pickle)

        else:
            conll_dataset = load_dataset(path=WordCountDataModule.ds_name,
                data_dir=WordCountDataModule.data_dir)
            embedder = VECTORS(cache=VECTORS_DIR)

            counts = defaultdict(int)
            for split in 'train', 'test', 'validation':
                for example in conll_dataset[split]:
                    for token in example['tokens']:
                        counts[token.lower()] += 1
            word_count_ds = []
            for t, c in counts.items():
                word_count_ds.append((embedder[t].float(), torch.tensor([log(c)], dtype=torch.float)))

            with open(WordCountDataModule.ds_cache, 'wb') as ds_pickle:
                pickle.dump(word_count_ds, ds_pickle)
        return word_count_ds


class WikiDataModule(HuggingfaceDataModule):
    ds_name = "wikicorpus"
    ds_cache = CACHE_DIR / 'wiki.pickle'
    data_dir = CACHE_DIR / 'wiki_data'

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def download_and_preprocess():
        conll_dataset = load_dataset(path=WikiDataModule.ds_name, name='raw_en',
            data_dir=WikiDataModule.data_dir)

if __name__== "__main__":
    ds = WikiDataModule.download_and_preprocess()

    # DS_NAME = "conll2003"

    # counts = defaultdict(int)
    # ds = load_dataset(DS_NAME)
    # for split in 'train', 'test', 'validation':
    #     for example in tqdm(ds[split]):
    #         for token in example['tokens']:
    #             counts[token.lower()] += 1

    # log_counts = defaultdict(int)
    # for w, count in counts.items():
    #     log_counts[w] = log(count)
    # sorted_counts = sorted(log_counts.values(), reverse=True)
    # df = pd.DataFrame(data=sorted_counts)
    # df.index = log(df.index)

    # df.plot()
    # plt.xlabel("sorted items in log scale")
    # plt.ylabel("frequency in log scale")
    # plt.legend([DS_NAME])
    # plt.show()