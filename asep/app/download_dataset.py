# basic
import argparse
import json
import os
import os.path as osp
import re
import shutil
import sys
import textwrap
import time
from argparse import Namespace
from pathlib import Path
from pprint import pprint
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Set, Tuple, Union)

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from asep.data.asepv1_dataset import AsEPv1Dataset

# ==================== Configuration ====================


# ==================== Function ====================


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Filter out problematic AbM numbered file identifiers.",
        epilog=textwrap.dedent(
            """
        Example usage:
            python download_dataset.py /path/to/save/dataset AsEP
        """
        ),
    )
    parser.add_argument("root", type=str, help="Root directory to save the dataset.")
    parser.add_argument(
        "dataset_name",
        type=str,
        default="AsEP",
        help="Name of the dataset to download.",
    )
    args = parser.parse_args()

    return args


def main(args):
    # Download the dataset with default embedding configuration
    AsEPv1Dataset(root=args.root, name=args.dataset_name)


def app():
    main(cli())
