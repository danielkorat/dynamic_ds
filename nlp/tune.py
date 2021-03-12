from model import WordCountPredictor
from dataset import WikiBigramsDataModule

import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


def train_word_count_tune(config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    model = WordCountPredictor(config=config)
    dm = WikiBigramsDataModule(config=config)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "val_loss",
                },
                on="validation_end")
        ])
    trainer.fit(model, datamodule=dm)


def train_word_count_tune_checkpoint(config,
                                checkpoint_dir=None,
                                num_epochs=10,
                                num_gpus=0):
    dm = WikiBigramsDataModule(config=config)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "val_loss",
                },
                filename="checkpoint",
                on="validation_end")
        ])
    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = Lightningword_countClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = WordCountPredictor._load_model_state(
            ckpt, config=config)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = WordCountPredictor(config=config)

    trainer.fit(model, datamodule=dm)

def tune_word_count_asha(num_samples=10, num_epochs=10, gpus_per_trial=0):
    WikiBigramsDataModule.download_and_preprocess()

    config = {
        'hidden_dim': tune.choice([32, 64, 128]),
        'learning_rate': tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        'batch_size': tune.choice([8, 16, 32, 64, 128]),
        'dropout_prob': tune.choice([0.0, 0.3]),
        # 'num_workers': 1
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["hidden_dim", "learning_rate", "batch_size", 'dropout_prob'],
        metric_columns=["loss", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_word_count_tune,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_word_count_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


def tune_word_count_pbt(num_samples=10, num_epochs=10, gpus_per_trial=0):
    WikiBigramsDataModule.download_and_preprocess()

    config = {
        'hidden_dim': tune.choice([32, 64, 128]),
        'learning_rate': 1e-4,
        'batch_size': 32,
        'dropout_prob': tune.choice([0.0, 0.3]),
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "learning_rate": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
            "batch_size": [8, 16, 32, 64, 128],
        })

    reporter = CLIReporter(
        parameter_columns=["hidden_dim", "learning_rate", "batch_size", 'dropout_prob'],
        metric_columns=["loss", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_word_count_tune_checkpoint,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_word_count_pbt")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()

    if args.smoke:
        tune_word_count_asha(num_samples=1, num_epochs=3, gpus_per_trial=0)
        tune_word_count_pbt(num_samples=1, num_epochs=3, gpus_per_trial=0)
    else:
        # ASHA scheduler
        tune_word_count_asha(num_samples=10, num_epochs=50, gpus_per_trial=0)
        # Population based training
        tune_word_count_pbt(num_samples=10, num_epochs=50, gpus_per_trial=0)
