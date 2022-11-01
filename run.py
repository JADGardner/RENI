# %%
def run_from_ipython():
    try:
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        return False

import os
from src.lightning.RENI_module import RENI
from src.lightning.callbacks import (
    LogExampleImagesCallback,
    MultiResTrainingCallback,
)
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import argparse

from types import SimpleNamespace

from configs.default import get_cfg_defaults


def main(config):

    #### LOGGER ####
    if config.TRAINER.LOGGER_TYPE == "wandb":
        logger = WandbLogger(
            name=config.TRAINER.LOGGER.WANDB.NAME,
            project=config.TRAINER.LOGGER.WANDB.PROJECT,
            save_dir=config.TRAINER.LOGGER.WANDB.SAVE_DIR,
            offline=config.TRAINER.LOGGER.WANDB.OFFLINE,
            log_model=config.TRAINER.LOGGER.WANDB.LOG_MODEL,
            config=config,
        )
    elif config.TRAINER.LOGGER_TYPE == "tensorboard":
        save_dir = config.TRAINER.LOGGER.TB.SAVE_DIR
        if config.TRAINER.LOGGER.TB.NAME == 'auto':
          name = f'latent_dim_{config.RENI.LATENT_DIMENSION}_net_' + \
                 f'{config.RENI.HIDDEN_LAYERS}_{config.RENI.HIDDEN_FEATURES}_' + \
                 f'{"vad" if config.RENI.MODEL_TYPE == "VariationalAutoDecoder" else "ad"}_' + \
                 f'{"cbc" if config.RENI.CONDITIONING == "Cond-by-Concat" else "film"}_' + \
                 f'{config.RENI.OUTPUT_ACTIVATION}_' + \
                 f'{"hdr" if config.DATASET[config.DATASET.NAME].IS_HDR else "ldr"}'
        else:
          name = config.TRAINER.LOGGER.TB.NAME
        logger = TensorBoardLogger(
            save_dir=save_dir,
            name=name,
            log_graph=config.TRAINER.LOGGER.TB.LOG_GRAPH,
        )
        # create the folder if it does not exist
        if not os.path.exists(save_dir + os.sep + name):
            os.makedirs(save_dir + os.sep + name)

    seed_everything(42, workers=True)

    precision = 16 if config.TRAINER.MIXED_PRECISION else 32

    assert config.RENI.TASKS[0] == "FIT_DECODER" if len(config.RENI.TASKS) > 1 and config.TRAINER.CHKPTS.LOAD_PATH is None else True
    if config.RENI.TASKS[0] != "FIT_DECODER":
        assert config.TRAINER.CHKPTS.LOAD_PATH is not None

    chkpt_path = config.TRAINER.CHKPTS.LOAD_PATH

    for task in config.RENI.TASKS:
        #### MODEL ####
        if chkpt_path is None:
            model = RENI(config=config, task=task)
        else:
            model = RENI.load_from_checkpoint(chkpt_path, config=config, task=task)

        #### CALLBACKS ####
        checkpoint_callback = ModelCheckpoint(
            save_top_k=2,
            every_n_epochs=config.TRAINER.CHKPTS.EVERY_N_EPOCHS,
            monitor=f"{task.lower()}_loss",
            filename=f"{task.lower()}_{{epoch:02d}}",
        )

        callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

        if config.TRAINER.LOGGER.LOG_IMAGES:
            callbacks.append(LogExampleImagesCallback())
        if config.RENI[task].MULTI_RES_TRAINING:
            callbacks.append(MultiResTrainingCallback())

        #### TRAINING ####
        if run_from_ipython():
            strategy = "ddp_notebook_find_unused_parameters_false"
        else:
            strategy = DDPStrategy(find_unused_parameters=False)

        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=config.RENI[task].EPOCHS,
            accelerator="auto",
            devices="auto",
            deterministic=True,
            strategy=strategy,
            precision=precision,
        )

        trainer.fit(model=model)

        if task == "FIT_DECODER":
            chkpt_path = checkpoint_callback.best_model_path

        if trainer.interrupted:
            break


# %%
if __name__ == "__main__":
    
    if run_from_ipython():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
        # Set CLI arguments here when using IPython in VSCode
        config = {"cfg_path": "configs/experiment.yaml"}
        args = SimpleNamespace(**config)

    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg_path', type=str, default='configs/experiment.yaml')
        parser.add_argument('--gpus', type=str, default='0, 1, 2, 3')
        args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    config = get_cfg_defaults()
    config.merge_from_file(args.cfg_path)
    main(config)