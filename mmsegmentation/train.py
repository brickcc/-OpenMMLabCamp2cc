import numpy as np

import os.path as osp
from tqdm import tqdm

import mmcv
import mmengine
from mmengine import Config

cfg = Config.fromfile('pspnet-Watermelon.py')

from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)


runner.train()