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

from argparse import ArgumentParser
from os import X_OK
from pprint import pprint

import torch
from torch import nn

import pytorch_lightning as pl

from nlp.data import WordCountDataModule, WikiDataModule
import numpy as np


class WordCountPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["learning_rate"]
        self.criterion = nn.MSELoss()
        self.embed_size = 100  # CharNGram embed size

        self.l1 = nn.Linear(self.embed_size, config["hidden_dim"])
        self.dropout = nn.Dropout(p=config["dropout_prob"])
        self.l2 = nn.Linear(config["hidden_dim"], 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.dropout(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("ptl/val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("ptl/test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def predict(model, dm, dl):

    Y_pred = []
    Y = []
    X = []
    for i, item in enumerate(dl):
        y = item[1][0]
        Y.append(y)
        original_x = dm.ds[dl.indices[i]]
        X.append(original_x)

        x_to_pred = item[0]
        Y_pred.append(model(x_to_pred).cpu().detach().numpy().flatten()[0])

    Y = np.round(np.exp(Y)).astype(int)
    Y_pred = np.round(np.exp(Y_pred)).astype(int)
    return X, Y, Y_pred


def dump(input, output, name):
    np.savez(name, x=input, y=output)


def train_simple_model(ds_name):
    pl.seed_everything(1234)

    config = {
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "dropout_prob": 0.0,
    }

    if ds_name =='conll2003':
        datamodule = WordCountDataModule(config)
    elif ds_name =='wikicorpus':
        datamodule = WikiDataModule(config)
    else:
        raise AssertionError(f'no dataset called {ds_name}')


    model = WordCountPredictor(config)

    # ------------
    # training
    # ------------

    trainer_args = {
        # 'gpus': 4,
        # 'accelerator': 'ddp',
        "max_epochs": 3,
    }
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, datamodule=datamodule)

    # ------------
    # testing
    # ------------
    # todo: without passing model it fails for missing best weights
    # MisconfigurationException, 'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.'
    test_loss = trainer.test(model, datamodule=datamodule)
    pprint(test_loss)

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

    paths = [f"true_{datamodule.ds_name}_test.npz",
             f"true_{datamodule.ds_name}_valid.npz",
             f"true_{datamodule.ds_name}_train.npz"]
    print(f"dumping test train and validation to:\n{' '.join(paths)}")

    dump(test_input, test_true, paths[0])
    dump(valid_input, valid_true, paths[1])
    dump(train_input, train_true, paths[2])

    filename = f"pred_{datamodule.ds_name}.npz"
    print(f"dumping test train and validation predictions to:\n{filename}")

    np.savez(
        filename,
        test_input=test_input,
        valid_input=valid_input,
        train_output=train_output,
        valid_output=valid_output,
        test_output=test_output,
        test_loss=test_loss,
    )


if __name__ == "__main__":
    train_simple_model('conll2003')
