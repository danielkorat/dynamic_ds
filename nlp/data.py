from http.client import RESET_CONTENT
from operator import itemgetter
from os.path import dirname, realpath, isfile
import pickle
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

import numpy as np
from nltk.lm import NgramCounter
from nltk.util import ngrams

CACHE_DIR = Path(dirname(realpath(__file__))) / "data"
VECTORS_DIR = CACHE_DIR / ".vector_cache"

VECTORS = CharNGram


class HuggingfaceDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.num_workers = 88  # config['num_workers']

    def prepare_data(self):
        # download only
        self.download_and_preprocess()

    @staticmethod
    @abstractmethod
    def download_and_preprocess():
        raise NotImplementedError()

    def setup(self, stage):
        ds = self.download_and_preprocess()
        self.ds = [i[2] for i in ds]
        ds = [i[:2] for i in ds]
        # Splits
        split_sizes = [int(len(ds) * 0.7), int(len(ds) * 0.15), int(len(ds) * 0.15)]
        if sum(split_sizes) < len(ds):
            split_sizes[0] += len(ds) - sum(split_sizes)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(ds, split_sizes)

    def split_dataloader(self, split):
        return DataLoader(
            split, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def train_dataloader(self):
        return self.split_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.split_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.split_dataloader(self.test_dataset)


class WordCountDataModule(HuggingfaceDataModule):
    ds_name = "conll2003"
    ds_cache = CACHE_DIR / "word_count.pickle"

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def download_and_preprocess():
        if isfile(WordCountDataModule.ds_cache):
            with open(WordCountDataModule.ds_cache, "rb") as ds_pickle:
                word_count_ds = pickle.load(ds_pickle)

        else:
            conll_dataset = load_dataset(
                path=WordCountDataModule.ds_name, cache_dir=CACHE_DIR
            )
            embedder = VECTORS(cache=VECTORS_DIR)

            counts = defaultdict(int)
            for split in "train", "test", "validation":
                for example in conll_dataset[split]:
                    for token in example["tokens"]:
                        counts[token.lower()] += 1
            word_count_ds = []
            for t, c in counts.items():
                word_count_ds.append(
                    (embedder[t].float(), torch.tensor([log(c)], dtype=torch.float), t)
                )

            with open(WordCountDataModule.ds_cache, "wb") as ds_pickle:
                pickle.dump(word_count_ds, ds_pickle)
        return word_count_ds


class WikiBigramsDataModule(HuggingfaceDataModule):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def download_and_preprocess(n=2, limit_prop=0.01):
        save_name = CACHE_DIR / f"{n}_grams_wikicorpus_{limit_prop * 100}%.npz"
        if isfile(save_name):
            print("Loading WikiBigramsDataModule from cache...")
            loaded = np.load(save_name)
            x, y = loaded['x'], loaded['y']
        else:
            VECTORS(cache=VECTORS_DIR)
            
            print("Saving WikiBigramsDataModule to cache...")
            x, y = save_ngram_counts('wikicorpus', limit_prop=limit_prop, n=n, tokens_key='sentence', 
                name='tagged_en', save_name=save_name)
        return x, y


def plot_frequencies(path, xlabel, ylabel, legend, save_name):
    data = np.load(path)
    counts = data['y']
    counts = np.log(np.flip(np.sort(counts)))
    print(f'Num of unique tokens: {len(counts)}')
    df = pd.DataFrame(data=counts)
    df.index = log(df.index + 1) # no log 0
    ds_name = path.rsplit( ".", 1 )[0].rsplit('/')[-1]
    fig = df.plot().get_figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([legend])
    fig.savefig(f'nlp/log_frequency_{save_name}.png')


def save_ngram_counts(ds_name, limit_prop, save_name, n=2, tokens_key='tokens', **kwargs):
    ds = load_dataset(ds_name, cache_dir=CACHE_DIR, **kwargs)['train']
    limit = int(limit_prop * len(ds))

    print('Computing n-grams...')
    n_grams = []
    for i, s in tqdm(enumerate(ds), total=limit):
        if i == limit:
            break
        n_grams.append(ngrams(s[tokens_key], n))
    del ds

    print('Counting n-grams...')
    res = {}
    for a, b_list in tqdm(NgramCounter(n_grams)[n].items()):
        for b, cnt in b_list.items():
            res[(a[0], b)] = cnt

    del n_grams
    xs, ys = [], []
    for (a, b), count in sorted(res.items(), key=itemgetter(1), reverse=True):
        xs.append(' '.join((str(a), str(b))))
        ys.append(count)

    print(f'Number of examples used: {limit}')
    print(f'Number of bigrams: {len(res)}')
    del res
    x_array, y_array = np.array(xs), np.array(ys)
    np.savez_compressed(save_name, x=x_array, y=y_array)
    return x_array, y_array

            
if __name__== "__main__":
    # plot_frequencies('wikicorpus', limit_prop=0.1, tokens_key='sentence', name='tagged_en')

    print(WikiBigramsDataModule.download_and_preprocess(limit_prop=0.01)[0][0])
