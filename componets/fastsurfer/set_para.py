from typing import Tuple, Union, Literal, Dict, Any, Optional
import glob
import sys
import argparse
import os
import copy

import numpy as np
# import torch
# import nibabel as nib
import torch
import pandas as pd
import sys
from .defaults import get_cfg_defaults
from .load_config import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def read_classes_from_lut(lut_file):
    """
    Function to read in FreeSurfer-like LUT table
    :param str lut_file: path and name of FreeSurfer-style LUT file with classes of interest
                         Example entry:
                         ID LabelName  R   G   B   A
                         0   Unknown   0   0   0   0
                         1   Left-Cerebral-Exterior 70  130 180 0
    :return pd.Dataframe: DataFrame with ids present, name of ids, color for plotting
    """
    # Read in file
    separator = {"tsv": "\t", "csv": ",", "txt": " "}
    return pd.read_csv(lut_file, sep=separator[lut_file[-3:]])


def load_config(cfg_file):
    # setup base
    cfg = get_cfg_defaults()
    cfg.EXPR_NUM = "Default"
    cfg.SUMMARY_PATH = ""
    cfg.CONFIG_LOG_PATH = ""
    # Overwrite with stored arguments
    cfg.merge_from_file(cfg_file)
    return cfg

def set_up_cfgs(cfg):
    cfg = load_config(cfg)
    # cfg.OUT_LOG_DIR = args.out_dir if args.out_dir is not None else cfg.LOG_DIR
    cfg.OUT_LOG_NAME = "fastsurfer"
    # cfg.TEST.BATCH_SIZE = args.batch_size

    cfg.MODEL.OUT_TENSOR_WIDTH = cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_HEIGHT = cfg.DATA.PADDED_SIZE
    return cfg


def args2cfg(path):
    """
    Extract the configuration objects from the arguments.
    """
    # cfg_cor = set_up_cfgs(args.cfg_cor, args) if args.cfg_cor is not None else None
    cfg_sag = set_up_cfgs(path)
    # cfg_ax = set_up_cfgs(args.cfg_ax, args) if args.cfg_ax is not None else None
    # cfg_fin = cfg_cor if cfg_cor is not None else cfg_sag if cfg_sag is not None else cfg_ax
    return cfg_sag
