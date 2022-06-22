The code is based on the PyTorch templated from https://github.com/vitusbenson/earthnet-pytorch-template
(also linked to the EarthNet repository: https://github.com/earthnet2021/earthnet-model-intercomparison-suite/tree/main/src/models).

## Setup:
1. Prepare the conda environment: `conda env create -f enpt111py39.yml`
2. Activate the environment: `conda activate enpt111py39`
3. Download the data (~500Gb): `python scripts/data_download.py`

## Training/testing/evaluation:
1. Set the desired parameters in a config file, e.g. `./src/models/pt_convlstm/configs/conv_lstm.yaml`
2. Train a model:
```
python ./src/models/pt_convlstm/train.py \
--setting="./src/models/pt_convlstm/configs/conv_lstm.yaml"
```
3. Predict on a test set:
```
python ./src/models/pt_convlstm/test.py \
--setting=./data/experiments/conv_lstm/version_0/settings.yaml \
--checkpoint=./data/experiments/conv_lstm/version_0/checkpoints/Epoch-epoch=58-ENS-EarthNetScore=0.3247.ckpt \
--track=iid \
--pred_dir=./data/scratch/preds/conv_lstm/version_0/iid_test_split
```

4. Compute the evaluation metrics:
```
python ./src/models/pt_convlstm/evaluate.py \
--dataroot="./data/en21ds_full" \
--outpath="./data/scratch/preds" \
--evalpath="./data/results" \
--model_name="conv_lstm" \
--split_name="iid_test_split" \
--experiment_name="version_0" \
--n_workers=16
```