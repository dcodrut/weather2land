import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import yaml

# go to the project root directory and add it to path
proj_root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(proj_root_dir))
os.chdir(proj_root_dir)
print(f"cwd = {Path.cwd()}")

from model.conv_lstm.conv_lstm_en import ConvLSTMen
from task.data import EarthNet2021DataModule
from task.stf import STFTask

__MODELS__ = {
    'conv_lstm': ConvLSTMen,
}

# https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
if 'SLURM_NTASKS' in os.environ:
    del os.environ['SLURM_NTASKS']
if 'SLURM_JOB_NAME' in os.environ:
    del os.environ['SLURM_JOB_NAME']


def train_model(setting_dict: dict):
    pl.seed_everything(setting_dict["Seed"])

    print(setting_dict)

    # Data
    data_args = ["--{}={}".format(key, value) for key, value in setting_dict["Data"].items()]
    data_parser = ArgumentParser()
    data_parser = EarthNet2021DataModule.add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = EarthNet2021DataModule(data_params)

    # Model
    model_args = ["--{}={}".format(key, value) for key, value in setting_dict["Model"].items()]
    model_parser = ArgumentParser()
    model_parser = __MODELS__[setting_dict["Architecture"]].add_model_specific_args(model_parser)
    model_params = model_parser.parse_args(model_args)
    model = __MODELS__[setting_dict["Architecture"]](model_params)

    # Task
    task_args = ["--{}={}".format(key, value) for key, value in setting_dict["Task"].items()]
    task_parser = ArgumentParser()
    task_parser = STFTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = STFTask(model=model, hparams=task_params)

    # Logger
    logger = pl.loggers.TensorBoardLogger(**setting_dict["Logger"])

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='EarthNetScore',
        filename='Epoch-{epoch:02d}-ENS-{EarthNetScore:.4f}',
        save_top_k=10,
        mode='max',
        every_n_epochs=1
    )
    summary = pl.callbacks.ModelSummary(max_depth=-1)

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["precision"] = 16 if dm.hparams.fp16 else 32
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback, summary], **trainer_dict)

    # Save the config file
    path_out = Path(logger.log_dir, 'settings.yaml')

    # Add the slurm job id if exists
    setting_dict_save = setting_dict.copy()
    setting_dict_save['SLURM_JOBID'] = os.environ['SLURM_JOBID'] if 'SLURM_JOBID' in os.environ else None
    print(f"Saving all the settings to {path_out}")
    path_out.parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, 'w') as fp:
        yaml.dump(setting_dict_save, fp, sort_keys=False)

    trainer.fit(task, dm)
    print(f"Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--setting', type=str, metavar='path/to/setting.yaml', help='yaml with all settings')
    parser.add_argument('--seed', type=int, help='training seed (to overwrite the one from the yaml config)',
                        default=None, required=False)
    args = parser.parse_args()

    with open(args.setting, 'r') as fp:
        setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

    # overwrite the seed if provided
    if args.seed is not None:
        setting_dict['Seed'] = args.seed

    train_model(setting_dict)
