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

from dataset import WordCountDataModule, WikiBigramsDataModule, WikiBigramsAdditionDataModule, plot_roc
import numpy as np

DATAMODULES = {
    'conll2003': WordCountDataModule,
    'wikicorpus': WikiBigramsDataModule,
    'wikicorpus_add': WikiBigramsAdditionDataModule
}


class WordCountPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["learning_rate"]
        self.criterion = nn.MSELoss()
        self.embed_size = 200 if config['concat'] else 100 # CharNGram embed size
        self.optim = config["optim"]

        self.l1 = nn.Linear(self.embed_size, config["hidden_dim"])
        self.dropout = nn.Dropout(p=config["dropout_prob"])
        self.l2 = nn.Linear(config["hidden_dim"], 1)

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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss)
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


def train_simple_model(ds_name, config=dict(), args=dict()):
    hpramas_str = '\n'.join(['\n', 'HYPERPARAMS', '-' * 11] + 
        [pformat({k: v}) for k, v in locals().items()] + ['\n'])
    print(hpramas_str)

    pl.seed_everything(123)

    datamodule = DATAMODULES[ds_name](config)
    
    model = WordCountPredictor(config)

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

    save_base = f"{'concat_' if config['concat'] else ''}{config['limit_prop'] * 100}%"
    base_path = Path(dirname(realpath(__file__)))
    targets_paths = {s: str(base_path / f"true_{datamodule.ds_name}_{s}_{save_base}.npz") \
        for s in ('train', 'test', 'valid')}

    print(f"dumping test train and validation to:\n{' '.join(targets_paths)}")
    
    dump(test_input, test_true, targets_paths['test'])
    dump(valid_input, valid_true, targets_paths['valid'])
    dump(train_input, train_true, targets_paths['train'])

    preds_path = str(base_path / f"pred_{datamodule.ds_name}_{save_base}.npz")
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

    return targets_paths, preds_path

if __name__ == "__main__":

    targets, preds = train_simple_model('wikicorpus_add', 
        config={
            "limit_prop": 0.03,
            "concat": False,
            'num_workers': 10,
            "hidden_dim": 128,
            "dropout_prob": 0.0,
            "optim": Adam,
            "learning_rate": 0.0001,
            "batch_size": 128
            },
        args={
            'gpus': 4,
            'max_epochs': 10
            })

    print(f"targets: {targets}")
    print(f"preds: {preds}")

    print('TEST ROC')
    plot_roc(targets=targets, preds=preds, split='test', hh_frac=0.01)

    # print('VAL ROC')
    # plot_roc(targets=targets, preds=preds, split='valid', hh_frac=0.01)
