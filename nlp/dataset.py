from operator import itemgetter
from os.path import dirname, realpath, isfile
import pickle
from collections import defaultdict
from abc import abstractmethod

import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from datasets import load_dataset
from numpy import log
from torchtext.vocab import CharNGram, FastText, GloVe
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from spacy.lang.en import English
from string import punctuation as punct

import numpy as np
from nltk.lm import NgramCounter
from nltk.util import ngrams

NLP_DIR = Path(dirname(realpath(__file__)))
CACHE_DIR = NLP_DIR / "data"
VECTORS_DIR = CACHE_DIR / ".vector_cache"

VECTORS = {
    'CharNGram': CharNGram,
    'FastText': FastText,
    'Glove': GloVe
}

FIXED_DIM_EMBED_TYPES = {
    'CharNGram': 100
}

EMBED_KWARGS ={
    'Glove': {'name': '6B'},
}

DS_KWARGS ={
    'wikicorpus': {'tokens_key': 'sentence', 'name': 'tagged_en'},
    'conll2003': {'tokens_key': 'tokens'}
}

class HuggingfaceDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # download only
        self.download_and_preprocess(config=self.config)

    @staticmethod
    @abstractmethod
    def download_and_preprocess(**kwargs):
        raise NotImplementedError()

    def setup(self, stage):
        ds = self.download_and_preprocess(config=self.config)
        self.ds = [i[2] for i in ds]
        ds = [i[:2] for i in ds]
        # Splits
        split_sizes = [int(len(ds) * p) for p in (0.4, 0.3, 0.3)]
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


def get_ds_repr(config):
    return f"{config['limit_prop'] * 100}%_{config['ds_name']}_{config['n']}-grams"

def get_feats_repr(config):
    return f"{get_ds_repr(config)}_{config['op']}_{config['embed_type']}.{config['embed_dim']}"

class NGramData(HuggingfaceDataModule):

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def download_and_preprocess(config, **kwargs):
        ds_name = config['ds_name']
        feats_repr = get_feats_repr(config)
        ngrams_path = CACHE_DIR / f"{get_ds_repr(config)}.npz"
        features_path = CACHE_DIR / f"{feats_repr}_features.npz" 

        if isfile(features_path):
            print(f"Loading features from: \n{features_path}")
            res = load_features_as_tensors(features_path)
        elif isfile(ngrams_path):
            print(f"Loading n-grams from: \n{ngrams_path}")
            loaded = np.load(ngrams_path)
            x, y = loaded['x'], loaded['y']
            res = save_features(x=x, y=y, cache=features_path, config=config)
        else:
            print(f"Saving n-grams to cache file: {ngrams_path}")    
            x, y = save_ngram_counts(cache=ngrams_path, config=config, **DS_KWARGS[ds_name])
            res = save_features(x=x, y=y, cache=features_path, config=config)
        return res

def plot_frequencies(y, xlabel, ylabel, save_name):
    counts = np.log(np.flip(np.sort(y)))
    
    df = pd.DataFrame(data=counts)
    df.index = log(df.index + 1) # no log 0
    # ds_name = path.rsplit( ".", 1 )[0].rsplit('/')[-1]
    fig = df.plot().get_figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([save_name])
    fig.savefig(NLP_DIR / f'{save_name}.png')

def plot_roc(targets: dict, preds: str, split: str, hh_frac=0.01):
    # plotting predictions for <percentile>-heavy hitter

    predictions = np.load(preds)
    true_data = np.load(targets[split])

    # print((predictions['test_input'] == true_data['x']).all()) # make sure test equals :)

    y_pred_scores = predictions[split + '_output']
    y_true_scores = true_data['y']

    threshold = np.flip(np.sort(y_true_scores))[
        int(y_true_scores.size * hh_frac)]

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

    plt.title(f'roc curve for {hh_frac}-heavy hitters model')
    plt.legend()

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    print('Saving plot...')
    plt.savefig(CACHE_DIR / f'roc_curve.png')
    print('Done.')

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

def save_ngram_counts(cache, config, tokens_key, **ds_kwargs):
    ds_name, limit_prop, n = config['ds_name'], config['limit_prop'], config['n']
    ds = load_dataset(ds_name, cache_dir=CACHE_DIR, **ds_kwargs)['train']
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

    print('Dumping ngrams to .npz...')
    np.savez_compressed(cache, x=x_array, y=y_array)
    return x_array, y_array

def save_features(x, y, cache, config):
    embed_type, embed_dim, op = config['embed_type'], config['embed_dim'], config['op']
    print(f"Saving features to cache file: {cache}")

    embedder_cls = VECTORS[embed_type]

    embed_kwargs = EMBED_KWARGS.get(embed_type, {})
    if embed_type in FIXED_DIM_EMBED_TYPES:
        embedder = embedder_cls(cache=VECTORS_DIR, **embed_kwargs)
    else:
        embedder = embedder_cls(cache=VECTORS_DIR, dim=embed_dim, **embed_kwargs)

    counts, embeds = [], []
    for bigram, count in tqdm(zip(x, y)):
        if op in ('concat', 'add'):
            word_a, word_b = bigram.split()
            embed_a = embedder[word_a]
            embed_b = embedder[word_b]

            bigram_embed = (embed_a + embed_b) if op == 'add' else \
                torch.cat((embed_a, embed_b), dim=-1)
        else:
            bigram_embed = embedder[bigram]

        embeds.append(bigram_embed.view(1, -1).cpu().detach().numpy().flatten())
        counts.append(log([count]))
    
    counts_arr, embeds_arr, bigrams_arr = np.array(counts), np.array(embeds), x
    np.savez_compressed(cache,
            counts=counts_arr, embeds=embeds_arr, bigrams=x)
    return counts_arr, embeds_arr, bigrams_arr

def load_features_as_tensors(features_path):
    feats = np.load(features_path)
    bigram_count_ds = []
    for count, embed, bigram in \
        tqdm(zip(feats['counts'], feats['embeds'], feats['bigrams'])):
        bigram_count_ds.append((torch.tensor(embed, dtype=torch.float),
            torch.tensor(count, dtype=torch.float), bigram))
    return bigram_count_ds
