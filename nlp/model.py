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

from data import WordCountDataModule
import numpy as np


class WordCountPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config['learning_rate']
        self.criterion = nn.MSELoss()
        self.embed_size = 100 # CharNGram embed size

        self.l1 = nn.Linear(self.embed_size, config['hidden_dim'])
        self.dropout = nn.Dropout(p=config['dropout_prob'])
        self.l2 = nn.Linear(config['hidden_dim'], 1)

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
        self.log('ptl/train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('ptl/val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('ptl/test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def predict(model, dl):
    output = []
    for item in dl:
        output.append(model(item[0]).cpu().detach().numpy().flatten())
    return output


def cli_main():
    pl.seed_everything(1234)

    config = {
        'hidden_dim': 128,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'dropout_prob': 0.0
    }

    datamodule = WordCountDataModule(config)
    model = WordCountPredictor(config)

    # ------------
    # training
    # ------------

    trainer_args = {
        # 'gpus': 4,
        # 'accelerator': 'ddp',
        'max_epochs': 3,
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
        test_output = predict(model, datamodule.test_dataloader())
        valid_output = predict(model, datamodule.val_dataloader())
        train_output = predict(model, datamodule.train_dataloader())
    filename =f'results.npz'

    np.savez(filename,
             train_output=train_output,
             valid_output=valid_output,
             test_output=test_output,
             test_loss=test_loss,
             )


if __name__ == '__main__':
    cli_main()