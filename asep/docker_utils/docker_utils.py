# basic
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Set, Tuple, Union)

import docker
import torch

USER=os.environ.get('USER')


# get host path for cwd (only works in devcontainer)
def _get_host_path_for_cwd() -> Path:
    '''
    When working in a devcontainer, we need to map the path in the container to the path on the host.
    This is because we need to pass the host path to run the docker container.
    e.g.
    - CWD in the devcontainer: /workspace/project_name/experiments/f1
    - LOCAL_WORKSPACE_FOLDER: /home/user/yum/ice/cream/project_name
    Returns: /home/user/yum/ice/cream/project_name/experiments/f1
    '''
    assert os.environ.get('LOCAL_WORKSPACE_FOLDER') is not None, 'LOCAL_WORKSPACE_FOLDER is not set'
    cwd = Path.cwd()
    local_workspace_folder = Path(os.environ['LOCAL_WORKSPACE_FOLDER']).resolve()
    # in case cwd is the same level as the local_workspace_folder on the host
    rel_path = '.' if len(p:=cwd.parts[:-cwd.parts[::-1].index(local_workspace_folder.name)]) == 0 else cwd.relative_to(*p)
    host_path = local_workspace_folder / rel_path
    return host_path

# docker
def run_igfold_docker_container(antibody_name: str, antibody_seqres: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run IgFold docker container to derive antibody SEQRES embeddings.

    Args:
        antibody_name (str): a unique name for the antibody
        antibody_seqres (Dict[str, str]): a dictionary of antibody SEQRES sequences

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: a tuple of the antibody embeddings and metadata
    """
    # Connect to Docker using the default socket or the configuration in your environment
    client = docker.from_env()

    # Run the container with the specified options
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(dir=cwd) as tmpdir:
        wd = tmpdir = Path(tmpdir)
        host_path = _get_host_path_for_cwd().joinpath(tmpdir.name).as_posix()
        # create a temporary folder with base folder as the workspace folder
        wd.joinpath('input').mkdir(parents=True, exist_ok=True)
        wd.joinpath('output').mkdir(parents=True, exist_ok=True)

        # IgFold
        # --------------------------------------------------------------------------
        # write seqres to p
        with open(wd/'input'/'ab.fasta', 'w') as f:
            for k, v in antibody_seqres.items():
                f.write(f">{antibody_name}|{k}\n{v}\n")

        # run docker
        container = client.containers.run(image=f"{USER}/igfold:runtime",
            command=["-i", "/home/vscode/input/ab.fasta",
                     "-o", "/home/vscode/output/",
                     "--embed"],
            volumes={
                f"{host_path}/input/": {'bind': '/home/vscode/input/', 'mode': 'rw'},
                f"{host_path}/output/": {'bind': '/home/vscode/output/', 'mode': 'rw'}
            },
            runtime="nvidia",
            detach=True,
            remove=False,  # if True => will remove the container after it stops
            tty=False,
            name=f"igfold-{antibody_name}")

        result = container.wait()  # wait for the container to finish
        log = container.logs()     # get the logs of the container
        metadata = {'result': result, 'log': log}
        # stop and remove container
        container.stop()
        container.remove()
        # --------------------------------------------------------------------------
        with open(wd/'output'/f'{antibody_name}.pkl', 'rb') as f:
            emb = pickle.load(f)
        return emb, metadata

# docker
def run_esm2_docker_container(antigen_name: str, antigen_seqres: Dict[str, str], esm2_ckpt_host_path: Union[Path, str], model_name: Optional[str]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(esm2_ckpt_host_path, str):
        esm2_ckpt_host_path = Path(esm2_ckpt_host_path)
    # Connect to Docker using the default socket or the configuration in your environment
    client = docker.from_env()

    # Run the container with the specified options
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(dir=cwd) as tmpdir:
        wd = tmpdir = Path(tmpdir)
        host_path = _get_host_path_for_cwd().joinpath(tmpdir.name)
        # create a temporary folder with base folder as the workspace folder
        wd.joinpath('input').mkdir(parents=True, exist_ok=True)
        wd.joinpath('output').mkdir(parents=True, exist_ok=True)

        # ESM2
        # --------------------------------------------------------------------------
        # write seqres to p
        with open(wd/'input'/'ag.fasta', 'w') as f:
            for k, v in antigen_seqres.items():
                f.write(f">{antigen_name}|{k}\n{v}\n")

        model_name="esm2_t12_35M_UR50D" if model_name is None else model_name
        n_layer=int(re.match(r"esm2_t(\d+)_\d+M_UR\d+D", model_name).group(1))

        # run docker
        container = client.containers.run(image=f"{USER}/esm2:runtime",
            command=[f"{model_name}",
                    "/home/vscode/esm2_input/ag.fasta",
                    "/home/vscode/esm2_output/",
                    "--toks_per_batch", "1024",
                    "--repr_layers", f"{n_layer}",
                    "--truncation_seq_length", "9999",
                    "--include", "per_tok"],
            volumes={
                f'{esm2_ckpt_host_path.as_posix()}/': {'bind': '/home/vscode/.cache/torch/hub/checkpoints/', 'mode': 'rw'},
                f'{host_path.as_posix()}/input/': {'bind': '/home/vscode/esm2_input/', 'mode': 'rw'},
                f'{host_path.as_posix()}/output/': {'bind': '/home/vscode/esm2_output/', 'mode': 'rw'}
            },
            runtime="nvidia",
            detach=True,
            remove=False,  # if True => will remove the container after it stops
            tty=False,
            name=f"esm2_{antigen_name}")

        result = container.wait()  # wait for the container to finish
        log = container.logs()     # get the logs of the container
        metadata = {'result': result, 'log': log}
        # stop and remove container
        container.stop()
        container.remove()
        # --------------------------------------------------------------------------
        pt = list(wd.joinpath('output').glob(f'{antigen_name}*.pt'))[0]
        emb = torch.load(pt)['representations'][n_layer]
        return emb, metadata
