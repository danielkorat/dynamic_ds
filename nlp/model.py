# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pprint import pformat

import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from os.path import dirname, realpath
from pathlib import Path

from pytorch_lightning.core.memory import ModelSummary
from nlp.dataset import NGramData, plot_roc, get_feats_repr
import numpy as np


class CountPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["learning_rate"]
        self.criterion = nn.MSELoss()
        self.optim = config["optim"]

        self.embed_size = config['embed_dim']
        if config['op'] == 'concat':
            self.embed_size = 2 * self.embed_size

        self.l1 = nn.Linear(self.embed_size, config["hidden_dim"])
        self.l2 = nn.Linear(config["hidden_dim"], 1)
        self.dropout = nn.Dropout(p=config["dropout_prob"])

    def step(self, batch):
        x, y = batch
        y_hat = self(x)
        return self.criterion(y_hat, y)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.dropout(x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.optim(params=self.parameters(), lr=self.lr)

def predict(model, dm, dl):
    Y_pred = []
    Y = []
    X = []
    for i, item in enumerate(dl):
        y = item[1][0]
        Y.append(y)
        original_x = dm.ds[dl.indices[i]]
        X.append(original_x)

        x_to_pred = item[0].reshape(-1, model.embed_size).to(model.device)
        Y_pred.append(model(x_to_pred).cpu().detach().numpy().flatten()[0])

    Y = np.round(np.exp(Y)).astype(int)
    Y_pred = np.round(np.exp(Y_pred)).astype(int)
    return X, Y, Y_pred


def dump(input, output, name):
    np.savez(str(name), x=input, y=output)

def train_simple_model(config=dict(), args=dict()):
    hpramas_str = '\n'.join(['\n', 'HYPERPARAMS', '-' * 11] + 
        [pformat({k: v}) for k, v in locals().items()] + ['\n'])
    print(hpramas_str)

    pl.seed_everything(123)
    datamodule = NGramData(config)
    model = CountPredictor(config)

    summary = ModelSummary(model, mode='full')
    model_size = summary.model_size
    print(f'Model_size: {model_size}')

    # ------------
    # training
    # ------------

    if 'gpus' in args:
        args['accelerator'] = 'ddp'

    trainer = pl.Trainer(**args)
    trainer.logger
    trainer.fit(model, datamodule=datamodule)

    # ------------
    # testing
    # ------------
    
    test_loss = trainer.test(model, datamodule=datamodule)

    print('Running prediction on all splits..')

    with torch.no_grad():
        test_input, test_true, test_output = predict(
            model, datamodule, datamodule.test_dataset
        )
        valid_input, valid_true, valid_output = predict(
            model, datamodule, datamodule.val_dataset
        )
        train_input, train_true, train_output = predict(
            model, datamodule, datamodule.train_dataset
        )

    feats_repr = get_feats_repr(config)
    base_path = Path(dirname(realpath(__file__)))
    targets_paths = {s: str(base_path / f"true_{feats_repr}_{s}.npz") \
        for s in ('train', 'test', 'valid')}

    print(f"dumping test train and validation to:\n{' '.join(targets_paths)}")
    
    dump(test_input, test_true, targets_paths['test'])
    dump(valid_input, valid_true, targets_paths['valid'])
    dump(train_input, train_true, targets_paths['train'])

    preds_path = str(base_path / f"pred_{feats_repr}.npz")
    print(f"dumping test train and validation predictions to:\n{preds_path}")

    np.savez(
        preds_path,
        test_input=test_input,
        valid_input=valid_input,
        train_output=train_output,
        valid_output=valid_output,
        test_output=test_output,
        test_loss=test_loss,
    )

    print(hpramas_str)
    print(f"targets: {targets_paths}")
    print(f"preds: {preds_path}")
    return targets_paths, preds_path, model_size

if __name__ == "__main__":
    config = config={
            'ds_name': 'wikicorpus',
            'embed_type': 'Glove',
            'embed_dim': 50,
            'op': 'concat',
            'n': 2,
            "limit_prop": 0.03,
            'num_workers': 22,
            "hidden_dim": 64,
            "dropout_prob": 0.0,
            "optim": Adam,
            "learning_rate": 0.001,
            "batch_size": 128
            }
    args= {
            # 'gpus': 4,
            'max_epochs': 1
            }

    targets, preds, model_size = train_simple_model(config=config, args=args)

    # print('TEST ROC')
    plot_roc(targets=targets, preds=preds, split='test', hh_frac=0.01)

    # print('VAL ROC')
    # plot_roc(targets=targets, preds=preds, split='valid', hh_frac=0.01)

    # targets = {'train': '/data/home/daniel_nlp/learning-ds/nlp/true_3.0%_wikicorpus_2-grams_concat_CharNGram.100_train.npz', 'test': '/data/home/daniel_nlp/learning-ds/nlp/true_3.0%_wikicorpus_2-grams_concat_CharNGram.100_test.npz', 'valid': '/data/home/daniel_nlp/learning-ds/nlp/true_3.0%_wikicorpus_2-grams_concat_CharNGram.100_valid.npz'}
    # preds = '/data/home/daniel_nlp/learning-ds/nlp/pred_3.0%_wikicorpus_2-grams_concat_CharNGram.100.npz'
    # plot_roc(targets=targets, preds=preds, split='test', hh_frac=0.01)