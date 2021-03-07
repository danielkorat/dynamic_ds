import torch
from torch import tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import random
from numpy import log
from torchtext.vocab import CharNGram

class WordCountDataModule(pl.LightningDataModule):
    def __init__(self, config, embedder_cls=CharNGram):
        super().__init__()
        self.batch_size = config['batch_size']
        self.lowercase = config['lowercase']
        self.embedder_cls = embedder_cls

    def add_unique_counts_and_embeds(self, ds):
        counts = defaultdict(int)
        embedder = self.embedder_cls()
        for split in 'train', 'test', 'validation':
            for example in tqdm(ds[split]):
                for token in example['tokens']:
                    counts[token.lower() if self.lowercase else token] += 1
        ds = []
        for t, c in counts.items():
            ds.append((embedder[t].float(), tensor([log(c)], dtype=torch.float)))
        return ds

    def prepare_data(self):
        # download only
        self.download_data()

        # ORIGINAL
        # # download only
        # MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        # MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    @staticmethod
    def download_data():
        CharNGram()
        load_dataset("conll2003")

    def setup(self, stage):
        conll_dataset = load_dataset("conll2003")
        ds = self.add_unique_counts_and_embeds(conll_dataset)
        split_sizes = [int(len(ds) * 0.7), int(len(ds) * 0.15), int(len(ds) * 0.15)]
        if sum(split_sizes) < len(ds):
            split_sizes[0] += len(ds) - sum(split_sizes)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(ds, split_sizes)

        # ## ORIGINAL
        # # # transform
        # transform = transforms.Compose([transforms.ToTensor()])
        # mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        # mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transform)

        # # train/val split
        # mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # # assign to use in dataloaders
        # self.train_dataset_orig = mnist_train
        # self.val_dataset = mnist_val
        # self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
