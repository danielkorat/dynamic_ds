from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import random
from numpy import log
import json


def gen_token_log_counts(dataset):
    token_counts = defaultdict(int)
    for split in 'train', 'test', 'validation':
        for element in tqdm(dataset[split]):
            for token in element['tokens']:
                token_counts[token.lower()] += 1
    return [(t, log(c)) for t, c in token_counts.items()]


def prepare_counts_dataset(log_counts, seed=42, splits=(0.7, 0.15, 0.15)):
    assert sum(splits) == 1.0
    random.seed(seed)
    random.shuffle(log_counts)
    split_1 = int(splits[0] * len(log_counts))
    split_2 = split_1 + int(splits[1] * len(log_counts))
    train = log_counts[:split_1]
    validation = log_counts[split_1: split_2]
    test = log_counts[split_2:]
    return {'train': train, 'validation': validation, 'test': test}


def main():
    conll_dataset = load_dataset("conll2003")
    log_counts = gen_token_log_counts(conll_dataset)
    counts_dataset = prepare_counts_dataset(log_counts)
    json.dump(counts_dataset, open('counts_dataset.json', 'w'))


if __name__ == "__main__":
    main()
