import json
import os
import shutil
import subprocess
import tempfile
import time
from argparse import Namespace
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from Bio import SeqIO
from loguru import logger
from pandas import DataFrame
from scipy.spatial.distance import cdist


def time_stamp():
    """generate a time stamp
    e.g. 20230611-204118
    year month day - hour minute second
    2006 06    11  - 20   41     18

    Returns:
        _type_: str
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ref: http://www.bioinf.org.uk/abs/info.html#cdrdef
CDR = {
    "ABM": {
        "H1": [26, 35],
        "H2": [50, 58],
        "H3": [95, 102],
        "L1": [24, 34],
        "L2": [50, 56],
        "L3": [89, 97],
    },
    "IMGT": {
        "H1": [26, 33],
        "H2": [51, 56],
        "H3": [93, 102],
        "L1": [27, 32],
        "L2": [50, 51],
        "L3": [89, 97],
    },
    "KABAT": {
        "H1": [31, 35],
        "H2": [50, 65],
        "H3": [95, 102],
        "L1": [24, 34],
        "L2": [50, 56],
        "L3": [89, 97],
    },
    "CHOTHIA": {
        "H1": [26, 32],
        "H2": [52, 56],
        "H3": [96, 101],
        "L1": [26, 32],
        "L2": [50, 52],
        "L3": [91, 96],
    },
}

_to_list = lambda cdr, a, b: [f"{cdr[0]}{i}" for i in range(a, b + 1)]
# 'H1' -> [H26, H26, ..., H35]
CDR2Resi = {
    name: {
        cdr: _to_list(
            cdr,
            a,
            b,
        )
        for cdr, (a, b) in d.items()
    }
    for name, d in CDR.items()
}
# 'H26' -> H1
Resi2CDR = {
    name: {resi: cdr for cdr, resis in d.items() for resi in resis}
    for name, d in CDR2Resi.items()
}


# helper: insert CRYST1 line to pdb file (required by DSSP)
def insert_cryst1_line_to_pdb(pdb_file: str) -> Optional[str]:
    """Insert a CRYST1 line to the pdb file if it doesn't exist, and return the new pdb file path."""
    pdb_file = Path(pdb_file)
    # check if CRYST1 line exists
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    if cryst1_line := [line for line in lines if line.startswith("CRYST1")]:
        print(f"CRYST1 line already exists in {pdb_file}")
        return pdb_file.as_posix()
    # add a CRYST1 line to the pdb file
    cryst1_line = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    # add the CRYST1 line before the first line startswith ATOM
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            lines.insert(i, cryst1_line)
            break
    # new file
    new_pdb_file = pdb_file.parent / f"{pdb_file.stem}_cryst1.pdb"
    with open(new_pdb_file, "w") as f:
        f.writelines(lines)  # write the new pdb file
    return new_pdb_file


# wrapper: call seqres2atmseq
def run_seqres2atmseq(seqres: str, atmseq: str) -> Dict[str, Any]:
    """
    Run seqres2atmseq to generate a mask file.
    Requires seqres2atmseq to be installed.
    ```
    $ pip install git+https://github.com/biochunan/seqres2atmseq.git
    ```

    Args:
        seqres (str): seqres sequence
        atmseq (str): atmseq sequence

    Raises:
        subprocess.CalledProcessError: if seqres2atmseq returns non-zero exit code

    Returns:
        Dict[str, Any]: a dictionary of the mask file
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        process = subprocess.Popen(
            [
                "seqres2atmseq",
                "-s",
                seqres,
                "-a",
                atmseq,
                "-o",
                Path(tmpdir) / "mask.json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        retcode = process.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, "seqres2atmseq", stderr)
        with open(Path(tmpdir) / "mask.json") as f:
            seqres2atmseq_mask = json.load(f)
    return seqres2atmseq_mask


# helper: extract SEQRES from a PDB file
def extract_seqres_from_pdb(struct_path: Path) -> Dict[str, str]:
    """
    Extract SEQRES from a PDB file.
    This function assumes SEQRES exists in the PDB file.

    Args:
        struct_path (Path): path to the PDB file

    Returns:
        Dict[str, str]: a dictionary of SEQRES sequences,
            chain_id (str) -> sequence (str)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_fp = struct_path
        if pdb_fp.suffix != ".pdb":
            pdb_fp = Path(tmpdir) / f"{pdb_fp.stem}_cryst1.pdb"
            shutil.copy(struct_path, pdb_fp)
        seqs = {}
        for record in SeqIO.parse(pdb_fp, "pdb-seqres"):
            c, s = record.annotations["chain"], str(record.seq)
            logger.info(f"chain {c}: {s}")
            if "X" in s:
                logger.warning(
                    f"Non-standard amino acid or HETATM (letter 'X') detected in chain {c} seqres: {record.id}"
                )
                # remove them
                # NOTE [chunan]: this might cause alignment error
                logger.info(
                    f"Removing non-standard amino acid or HETATM (letter 'X') from chain {c} seqres"
                )
            seqs[c] = s.replace("X", "")
        return seqs


# helper: calculate distance matrix
def calculate_dist_mat(
    df1: DataFrame, df2: DataFrame, coord_columns: Optional[List[str]] = None
) -> np.ndarray:
    """
    Calculate the distance matrix between two sets of coordinates.
    Assumes that the coordinates are in the columns 'x_coord', 'y_coord', 'z_coord'.

    Args:
        df1 (DataFrame): Structure DataFrame of object 1
        df2 (DataFrame): Structure DataFrame of object 2

    Returns:
        np.ndarray: Distance matrix between the two sets of coordinates
    """
    coord_columns = coord_columns or ["x_coord", "y_coord", "z_coord"]
    try:
        assert set(coord_columns).issubset(df1.columns)
        assert set(coord_columns).issubset(df2.columns)
    except AssertionError as e:
        raise ValueError(
            f"`coord_columns` must be a subset of columns in df1 and df2"
        ) from e

    return cdist(df1[coord_columns], df2[coord_columns])


# helper: log dict
def log_dict(d: Dict[str, Any], str_len: int = 20) -> None:
    for k, v in d.items():
        logger.info(f"{k:<{str_len}}: {v}")


# helper: log args
def log_args(args: Union[Namespace, Dict]) -> None:
    # log the file name
    logger.info(f"Input arguments:")
    if isinstance(args, Namespace):
        log_dict(args.__dict__, str_len=20)
    elif isinstance(args, Dict):
        log_dict(args, str_len=20)


# helper: convert devcontainer path to host path
def map_devcontainer_path_to_host_path(dev_path: Path) -> Path:
    """
    Map devcontainer path to host path.
    Requires the env var LOCAL_WORKSPACE_FOLDER to be set.
    Example:
    - dev_path: '/workspaces/project_name/experiments/walle'
    - local_workspace_folder: '/host/user/path/to/project_name'
    - rel_path: 'experiments/walle'
    - host_path: '/host/user/path/to/project_name/experiments/walle'

    Args:
        dev_path (Path): path inside the devcontainer
    Returns:
        Path: path on the host
    """
    # assert the env var LOCAL_WORKSPACE_FOLDER is set
    try:
        assert (p := os.environ.get("LOCAL_WORKSPACE_FOLDER")) is not None
    except AssertionError as e:
        raise ValueError("LOCAL_WORKSPACE_FOLDER is not set") from e

    local_workspace_folder = Path(p).resolve()

    # assert the mapped project folder name is in the dev_path
    try:
        assert local_workspace_folder.name in dev_path.parts
    except AssertionError as e:
        raise ValueError(f"{local_workspace_folder.name} is not in {dev_path}") from e

    # in case cwd is the same level as the local_workspace_folder on the host
    rel_path = (
        "."
        if len(
            p := dev_path.parts[
                : -dev_path.parts[::-1].index(local_workspace_folder.name)
            ]
        )
        == 0
        else dev_path.relative_to(*p)
    )
    host_path = local_workspace_folder / rel_path

    return host_path


# helper: convert any types that are not json serializable to str
def convert_to_json_serializable(data) -> Any:
    if isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().tolist()
    elif isinstance(data, set):
        return list(data)
    # Add other types as needed
    else:
        raise TypeError(f"Type {type(data)} not serializable")


# decorator: log function duration
def dec_log_func(func: Callable):
    @wraps(func)
    def wrapper(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        duration = te - ts
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        logger.info(f"{func.__name__} took: {minutes} min {seconds} sec")
        return result

    return wrapper
