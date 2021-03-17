#%%
"""
Runs a model on a single node across N-gpus.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

from pytorch_lightning.accelerators.accelerator import Accelerator

from classifier_mlabel import Classifier
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torchnlp.random import set_seed

from sklearn.metrics import classification_report
from loguru import logger
#%%




'''
The trivial solution to Pr = Re = F1 is TP = 0. So we know precision, recall and F1 can have the same value in general
'''


def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    set_seed(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------

    model = Classifier(hparams)
    

    version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    name=f'{hparams.encoder_model}'

    save_path = 'experiments/' + name + '/' + version
    tb_logger = TensorBoardLogger(
        save_dir='experiments/',
        version=version,
        name=name
    )

    # # Early stopping

    # early_stop = EarlyStopping(
    #     monitor=hparams.early_stop_metric,
    #     patience=hparams.early_stop_patience,
    #     verbose=True,
    #     mode=hparams.early_stop_mode
    # )

    # Checkpointing
    # hparams.model_save_path = 'experiments/'
    # model_save_path = '{}/{}/{}'.format(hparams.model_save_path, hparams.encoder_model, version)
    # checkpoint = ModelCheckpoint(
    #     filepath=model_save_path,
    #     # save_function=None,
    #     # save_best_only=True,
    #     verbose=True,
    #     monitor=hparams.monitor,
    #     # mode=hparams.model_save_monitor_mode
    # )
    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        fast_dev_run=hparams.fast_dev_run,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        default_root_dir=f'./classifier_pipeline/{hparams.encoder_model}',
        accelerator='dp',
        profiler="simple",
        # checkpoint_callback=checkpoint,
        overfit_batches=0.01,
        limit_train_batches=0.1,
        limit_val_batches=0.01
        # early_stop_callback=early_stop,
    )

    # ------------------------
    # 6 START TRAINING
    # ------------------------

    #datamodule = MedNLIDataModule
    trainer.fit(model, model.data)
    trainer.test(model, model.data.test_dataloader())

    # cms = np.array(model.test_conf_matrices)
    # np.save(f'experiments/{model.hparams.encoder_model}/test_confusion_matrices.npy',cms)
    preds = model.test_predictions.detach().cpu()
    target = model.test_labels.detach().cpu()
    logger.info(classification_report(preds, target))
    
    import pandas as pd
    decoded_preds = model.mlb.inverse_transform(preds)
    decoded_truth = model.mlb.inverse_transform(target)
    # out = pd.DataFrame.from_dict({"preds": preds, "truth": target, "decoded_preds": decoded_preds, "decoded_truth": decoded_truth})
    out = pd.DataFrame.from_dict({"decoded_preds": decoded_preds, "decoded_truth": decoded_truth})
    out.to_csv('model_output.csv')



if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )


    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )

    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )

    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    parser.add_argument(
        '--fast_dev_run',
        default=False,
        type=bool,
        help='Run for a trivial single batch and single epoch.'
    )
    parser.add_argument(
        '--mid_dev_run',
        default=False,
        type=bool,
        help='Run on 1000 samples.'
    )
    # Batching
    parser.add_argument(
        "--batch_size", default=6, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")

    parser.add_argument("--nn_arch", type=str, default='default')



    # each LightningModule defines arguments relevant to it
    parser = Classifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)

# # %%
# class hparams:
#     def __init__(self) -> None:
#         self.seed=3
#         self.save_top_k=1
#         self.monitor='vall_acc'
#         self.metric_mode='max'
#         self.patience=5
#         self.max_epochs=10
#         self.fast_dev_run=True
#         self.batch_size=12
#         self.accumulate_grad_batches=2
#         self.gpus=1

# #%%
# from types import SimpleNamespace

# sn = SimpleNamespace()
# sn.seed=3
# sn.save_top_k=1
# sn.monitor='vall_acc'
# sn.metric_mode='max'
# sn.patience=5
# sn.max_epochs=10
# sn.fast_dev_run=True
# sn.batch_size=12
# sn.accumulate_grad_batches=2
# sn.gpus=1


# main(sn)

# # %%
