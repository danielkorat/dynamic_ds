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
from pprint import pprint

import torch
from torch.nn import functional as F

import pytorch_lightning as pl

from nlp.data import CountsDataModule
import numpy as np

class CountPredictor(pl.LightningModule):

    def __init__(self, embed_size=100, hidden_dim=128, learning_rate=1e-3, criterion=torch.nn.MSELoss()):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = self.hparams.criterion

        self.l1 = torch.nn.Linear(self.hparams.embed_size, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.hparams.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.hparams.criterion(y_hat, y)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.hparams.criterion(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def predict(model, dl):
    output = []
    for item in dl:
        output.append(model(item[0]).cpu().detach().numpy().flatten())
    return output


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CountPredictor.add_model_specific_args(parser)
    parser = CountsDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = CountsDataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    model = CountPredictor()

    # ------------
    # training
    # ------------
    # args.backend = 'ddp2'
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    # todo: without passing model it fails for missing best weights
    # MisconfigurationException, 'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.'
    test_loss = trainer.test(model, datamodule=dm)
    pprint(test_loss)

    test = dm.test_dataloader()
    valid = dm.val_dataloader()
    train = dm.train_dataloader()

    with torch.no_grad():

        test_output = predict(model, test)
        valid_output = predict(model, valid)
        train_output = predict(model, train)
    filename =f'results.npz'

    np.savez(filename,
             train_output=train_output,
             valid_output=valid_output,
             test_output=test_output,
             test_loss=test_loss,
             )



if __name__ == '__main__':
    cli_main()