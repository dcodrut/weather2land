import os
from pathlib import Path
import earthnet as en

# go to the project root directory and add it to path
proj_root_dir = Path(__file__).parent.parent
os.chdir(proj_root_dir)
print(f"cwd = {Path.cwd()}")

en.Downloader.get('./data/en21ds_full', "iid")
en.Downloader.get('./data/en21ds_full', "train")
en.Downloader.get('./data/en21ds_full', "ood")
en.Downloader.get('./data/en21ds_full', "extreme")
en.Downloader.get('./data/en21ds_full', "seasonal")
