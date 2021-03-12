from http.client import RESET_CONTENT
from itertools import combinations
from operator import itemgetter
from os.path import dirname, realpath, isfile
import pickle
from collections import defaultdict
from abc import abstractmethod
from threading import current_thread

import torch
from sklearn.metrics import roc_auc_score, roc_curve
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

from spacy.lang.en import English
from string import punctuation as punct

class HuggingfaceDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        config_init = {'num_workers': 10}
        config_init.update(config)
        self.config = config_init

    def prepare_data(self):
        # download only
        self.download_and_preprocess(**self.config)

    @staticmethod
    @abstractmethod
    def download_and_preprocess(**kwargs):
        raise NotImplementedError()

    def setup(self, stage):
        ds = self.download_and_preprocess(**self.config)
        self.ds = [i[2] for i in ds]
        ds = [i[:2] for i in ds]
        # Splits
        split_sizes = [int(len(ds) * 0.7), int(len(ds) * 0.15), int(len(ds) * 0.15)]
        if sum(split_sizes) < len(ds):
            split_sizes[0] += len(ds) - sum(split_sizes)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(ds, split_sizes)

    def split_dataloader(self, split):
        return DataLoader(
            split, batch_size=self.config['batch_size'], num_workers=self.config['num_workers']
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
    def download_and_preprocess(**kwargs):
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
    ds_name = "wikicorpus"

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def download_and_preprocess(n=2, limit_prop=0.01, concat=False, **kwargs):
        save_name = f"{n}_grams_wikicorpus_{'concat_' if concat else ''}{limit_prop * 100}%"
        np_cache_file = CACHE_DIR / f"{save_name}.npz"
        features_cache = CACHE_DIR / f'{save_name}_features.npz'
        if isfile(np_cache_file) and isfile(features_cache):
            print("Loading WikiBigramsDataModule from cache...")
            loaded = np.load(np_cache_file)
            x, y = loaded['x'], loaded['y']
            loaded = np.load(features_cache)
            counts_arr, embeds_arr, bigrams_arr = loaded['counts'], loaded['embeds'], loaded['bigrams']
        else:
            embedder = VECTORS(cache=VECTORS_DIR)
            print("Saving WikiBigramsDataModule to cache...")
            x, y = save_ngram_counts('wikicorpus', limit_prop=limit_prop, n=n, tokens_key='sentence', 
                name='tagged_en', save_name=save_name)

            counts, embeds = [], []
            for bigram, count in zip(x, y):
                if concat:
                    word_a, word_b = bigram.split()
                    embed_a = embedder[word_a]
                    embed_b = embedder[word_b]
                    bigram_embed = torch.cat((embed_a, embed_b), dim=1).cpu().detach().numpy().flatten()
                else:
                    bigram_embed = embedder[bigram].cpu().detach().numpy().flatten()
                embeds.append(bigram_embed)
                counts.append(log([count]))
            
            counts_arr, embeds_arr, bigrams_arr = np.array(counts), np.array(embeds), x
            np.savez_compressed(features_cache,
                    counts=counts_arr, embeds=embeds_arr, bigrams=x)
        
        bigram_count_ds = []
        for count, embed, bigram in zip(counts_arr, embeds_arr, bigrams_arr):
            bigram_count_ds.append((torch.tensor(embed, dtype=torch.float),
                torch.tensor(count, dtype=torch.float), bigram))

        # plot_frequencies(y=y, xlabel='Sorted items in log scale',
        #                 ylabel='Frequency in log scale', save_name=save_name)

        return bigram_count_ds

def plot_frequencies(y, xlabel, ylabel, save_name):
    counts = np.log(np.flip(np.sort(y)))
    
    df = pd.DataFrame(data=counts)
    df.index = log(df.index + 1) # no log 0
    # ds_name = path.rsplit( ".", 1 )[0].rsplit('/')[-1]
    fig = df.plot().get_figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([save_name])
    fig.savefig(CACHE_DIR / f'{save_name}.png')


def plot_roc(pred_data_path, true_data_path,dump_path, predictions_key='test_output',
             hh_fraction=0.01):
    # plotting predictions for <percentile>-heavy hitter

    predictions = np.load(pred_data_path)
    true_data = np.load(true_data_path)

    # print((predictions['test_input'] == true_data['x']).all()) # make sure test equals :)

    y_pred_scores = predictions[predictions_key]
    y_true_scores = true_data['y']

    threshold = np.flip(np.sort(y_true_scores))[
        int(y_true_scores.size * hh_fraction)]

    y_true = np.zeros_like(y_true_scores)
    y_true[y_true_scores >= threshold] = 1

    ns_scores = [0] * len(y_true)

    ns_auc = roc_auc_score(y_true, ns_scores)
    lr_auc = roc_auc_score(y_true, y_pred_scores)

    print('No Skill: ROC AUC=%.2f' % (ns_auc))
    print('Learned: ROC AUC=%.2f' % (lr_auc))

    fpr, tpr, _ = roc_curve(y_true, y_pred_scores)
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_scores)

    plt.plot(fpr, tpr, marker='.', label=f'Learned model- AUC={lr_auc:.2f}')
    plt.plot(ns_fpr, ns_tpr, marker='.')

    plt.title(f'roc curve for {hh_fraction}-heavy hitters model')
    plt.legend()

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.savefig(dump_path)

def clean(toks: list, nlp):
    res_texts, cur_text = [], []
    for t in toks:
        if nlp.vocab[t].is_stop or any(c in punct for c in t):
            if cur_text:
                res_texts.append(cur_text)
                cur_text = []
        else:
            cur_text.append(t.lower())
            
    return res_texts


def save_ngram_counts(ds_name, limit_prop, save_name, n=2, tokens_key='tokens', **kwargs):
    ds = load_dataset(ds_name, cache_dir=CACHE_DIR, **kwargs)['train']
    limit = int(limit_prop * len(ds))

    print('Computing n-grams...')
    n_grams = []
    nlp = English()
    for i, s in tqdm(enumerate(ds), total=limit):
        if i == limit:
            break
        tokens = s[tokens_key]
        cleaned_texts = clean(tokens, nlp)
        for clean_text in cleaned_texts:
            n_grams.append(ngrams(clean_text, n))
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
    np.savez_compressed(CACHE_DIR / f'{save_name}.npz', x=x_array, y=y_array)
    return x_array, y_array

            
if __name__== "__main__":
    WikiBigramsDataModule.download_and_preprocess(limit_prop=0.001, concat=True)
