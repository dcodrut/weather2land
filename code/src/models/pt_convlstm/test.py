import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
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


def test_model(setting_dict: dict, checkpoint: str):
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
    task.load_from_checkpoint(checkpoint_path=checkpoint, context_length=setting_dict["Task"]["context_length"],
                              target_length=setting_dict["Task"]["target_length"], model=model, hparams=task_params)

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["precision"] = 16 if dm.hparams.fp16 else 32
    trainer = pl.Trainer(**trainer_dict)

    trainer.test(model=task, datamodule=dm, ckpt_path=None)


def get_best_model_ckpt(checkpoint_dir):
    ckpt_list = sorted(list(Path(checkpoint_dir).glob('*.ckpt')))
    ens_list = np.array([float(p.stem.split('EarthNetScore=')[1]) for p in ckpt_list])

    # get the index of the last maximum value
    i_max = len(ens_list) - np.argmax(ens_list[::-1]) - 1
    ckpt_best = ckpt_list[i_max]

    return ckpt_best


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--setting', type=str, metavar='path/to/setting.yaml', help='yaml with all settings')
    parser.add_argument('--checkpoint', type=str, metavar='path/to/checkpoint', help='checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str, metavar='path/to/checkpoint_dir',
                        help='a directory from which the model with the best ENS score will be selected '
                             '(alternative to checkpoint_file)', default=None)
    parser.add_argument('--track', type=str, metavar='iid|ood|ex|sea',
                        help='which track to test: either iid, ood, ex or sea')
    parser.add_argument('--pred_dir', type=str, default=None, metavar='path/to/predictions/directory/',
                        help='Path where to save predictions')
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        checkpoint_file = args.checkpoint
    else:
        checkpoint_file = get_best_model_ckpt(args.checkpoint_dir)
        print(f"Best checkpoint: {checkpoint_file}")

    with open(args.setting, 'r') as fp:
        setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

    setting_dict["Task"]["context_length"] = 10 if args.track in ["iid", "ood"] \
        else 20 if args.track == "ex" else 70 if args.track == "sea" else 10
    setting_dict["Task"]["target_length"] = 20 if args.track in ["iid", "ood"] \
        else 40 if args.track == "ex" else 140 if args.track == "sea" else 20
    setting_dict["Model"]["context_length"] = setting_dict["Task"]["context_length"]
    setting_dict["Model"]["target_length"] = setting_dict["Task"]["target_length"]

    setting_dict["Data"]["test_track"] = args.track

    if args.pred_dir is not None:
        setting_dict["Task"]["pred_dir"] = args.pred_dir

    test_model(setting_dict, checkpoint_file)
