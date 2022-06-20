import sys
import os
from pathlib import Path

# go to the project root directory and add it to path
proj_root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(proj_root_dir))
os.chdir(proj_root_dir)
print(f"cwd = {Path.cwd()}")

from earthnet.parallel_score import EarthNetScore
from options.base_options import BaseOptions

# parse arguments
opt = BaseOptions().parse(save=True)

'''
Evaluates results using src/evaluation/parallel_score.py
Loops over the lists of model_names and experiment_names given to run.py.
Saves results to results directory.
'''

# If test set is split into context/targets use subdirectory targets for GT
if 'split' in opt.split_name:
    targ_dir = os.path.join(opt.dataroot, opt.split_name, 'target')
else:
    targ_dir = os.path.join(opt.dataroot, opt.split_name)

for i, model_exp in enumerate(zip(opt.model_name, opt.experiment_name)):
    model, experiment = model_exp
    m_e = os.path.join(model, experiment)
    print("Model/Experiment ({0}) to evaluate is {1}".format(i, m_e))

    save_dir = os.path.join(opt.evalpath, 'logs', opt.split_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_output_file = os.path.join(save_dir, model, 'data', m_e.replace('/', '_') + '.json')
    ens_output_file = os.path.join(save_dir, model, 'ens', m_e.replace('/', '_') + '.json')
    pred_dir = os.path.join(opt.outpath, m_e, opt.split_name)

    EarthNetScore.get_ENS(pred_dir, targ_dir, n_workers=opt.n_workers, data_output_file=data_output_file,
                          ens_output_file=ens_output_file)
