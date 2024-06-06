'''
Requires biopandas to parse pdb file
pip install biopandas

Usage:

python map-highprob-vertices-to-residues.py \
    --job_name 1cz8_V \
    --data_preparation /path/to/masif_output/data_preparation \
    --output /path/to/masif_output/pred_output \
    --prob_thr 0.7 --radius 1.2

'''
import argparse
from pathlib import Path
from typing import List

import numpy as np
from biopandas.pdb import PandasPdb


def match_atoms(target_coords: np.ndarray, query_coords: np.ndarray, radius: float) -> List[int]:
    """
    Find atoms that are within a radius from the high-prob vertices

    Args:
        target_coords(np.ndarray): shape (N, 3), coordinates of the vertices
        query_coords (np.ndarray): shape (N, 3), coordinates of the atoms
        radius (float)           : radius in Å cutoff to retrieve atoms close to vertices

    Returns:
        idx (List[int]): indices of the atoms IN THE QUERY_COORDS
            that are within a radius from the vertices
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(query_coords)  # indexing the atom coordinates
    # get atoms that are within a radius from the vertices
    idx = tree.query_ball_point(target_coords, r=radius)
    # flatten the list of lists
    idx = [item for sublist in idx for item in sublist]
    # remove duplicates
    idx = list(set(idx))

    return idx


# input
def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--job_name", type=str, help="job name assigned to the inference run, e.g. 1cz8_V")
    parser.add_argument("--data_preparation", type=str, help="path to data_preparation folder")
    parser.add_argument("--output", type=str, help="path to the inference output folder")
    parser.add_argument("--prob_thr", type=float, default=0.7,
                        help="probability threshold to filter vertices")
    parser.add_argument("--radius", type=float, default=1.2,
                        help="radius in Å to retrieve atoms close to vertices")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = cli()

    # args = argparse.Namespace()
    # args.job_name = '1cz8_V'
    # args.data_preparation = Path('/home/chunan/UCL/scripts/Antibody-specific-epitope-prediction-2.0/experiments/masif-site/masif_output/data_preparation')
    # args.output = Path('/home/chunan/UCL/scripts/Antibody-specific-epitope-prediction-2.0/experiments/masif-site/masif_output/pred_output')
    # args.prob_thr = 0.7
    # args.radius = 1.2

    job_name = args.job_name
    inp_path = Path(args.data_preparation)
    out_path = Path(args.output)

    # --------------------
    # file paths
    # --------------------
    # pred probabilities of each vertex being part of a binding site
    pred_fp = out_path.joinpath('all_feat_3l', 'pred_data', f'pred_{job_name}.npy')
    # input pdb file to masif-site
    in_pdb_fp = inp_path.joinpath('01-benchmark_pdbs', f'{job_name}.pdb')

    # --------------------
    # read pred files
    # --------------------
    # Load the predicted surface numpy file.
    # If running inside a container, this is located by default under `/masif/data/masif_site/output/all_feat_3l/pred_surfaces`
    # e.g. `/masif/data/masif_site/output/all_feat_3l/pred_surfaces/1cz8_V.npy`
    probs = np.load(pred_fp)

    # Load vertex coordinates from precomputed files
    # If running in the container, this is located under `/masif/data/masif_site/data_preparation/04a-precomputation_9A/precomputation/`
    # e.g. `/masif/data/masif_site/data_preparation/04a-precomputation_9A/precomputation/1cz8_V`
    # - `1cz8_V` is the job name I used for this example.
    precompute_folder = inp_path.joinpath('04a-precomputation_9A', 'precomputation', job_name)

    # We only need the coordinates, other features are not needed in this notebook
    p1_X = np.load(precompute_folder/'p1_X.npy')
    p1_Y = np.load(precompute_folder/'p1_Y.npy')
    p1_Z = np.load(precompute_folder/'p1_Z.npy')
    # print(f'p1_X.shape:                {p1_X.shape}')
    # print(f'p1_Y.shape:                {p1_Y.shape}')
    # print(f'p1_Z.shape:                {p1_Z.shape}')


    # #  ---------- Other pre-computed features ----------
    # p1_iface_labels     = np.load(base/'p1_iface_labels.npy')
    # p1_feat             = np.load(base/'p1_input_feat.npy')
    # p1_list_indices     = np.load(base/'p1_list_indices.npy', allow_pickle=True)  # make sure to add allow_pickle=True
    # p1_mask             = np.load(base/'p1_mask.npy')
    # p1_rho_wrt_center   = np.load(base/'p1_rho_wrt_center.npy')
    # p1_theta_wrt_center = np.load(base/'p1_theta_wrt_center.npy')

    # # print the shape of each array
    # print(f'p1_iface_labels.shape:     {p1_iface_labels.shape}')
    # print(f'p1_feat.shape:             {p1_feat.shape}')
    # print(f'p1_list_indices.shape:     {p1_list_indices.shape}')
    # print(f'p1_mask.shape:             {p1_mask.shape}')
    # print(f'p1_rho_wrt_center.shape:   {p1_rho_wrt_center.shape}')
    # print(f'p1_theta_wrt_center.shape: {p1_theta_wrt_center.shape}')

    # --------------------
    # map high prob vertices to
    # residues in the structure
    # --------------------
    # use biopandas to parse the pdb file
    pdb = PandasPdb().read_pdb(in_pdb_fp)

    # convert to dataframe
    atom_df = pdb.df['ATOM']
    # add node_id in the format of [chain_id]:[residue_name]:[residue_number]:[insertion]
    atom_df['node_id'] = atom_df['chain_id'] + ':' + atom_df['residue_name'] + ':' + atom_df['residue_number'].astype(str) + ':' + atom_df['insertion']
    # remove the tailing space and colon in the node_id if insertion is empty
    atom_df['node_id'] = atom_df['node_id'].str.replace(r':\s*$', '', regex=True)

    # --------------------
    # Find atoms close to the
    # predicted surface vertices
    # --------------------
    # params
    prob_thr = args.prob_thr  # prob thr to filter vertices
    radius = args.radius      # radius in Å to retrieve atoms close to vertices

    # get the coordinates of the vertices (shape N_vertices, 3)
    vertices_coords = np.concatenate([p1_X.reshape(-1, 1),
                                      p1_Y.reshape(-1, 1),
                                      p1_Z.reshape(-1, 1)], axis=1)

    # get vertex with the probability greater than a threshold e.g. 0.9
    _, idx = np.where(probs > prob_thr)

    # get the coordinates of the vertices with the probability greater than a threshold
    hp_coords = vertices_coords[idx]

    # atom coordinates from the pdb file
    atom_coords = atom_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].to_numpy()

    # atom df row indices
    idx = match_atoms(hp_coords, atom_coords, radius)

    # pymol selection string
    resis = atom_df.iloc[idx].drop_duplicates(subset=['node_id']).node_id.map(lambda x: x.split(':')[-1]).unique()
    chain_id = atom_df.chain_id.values[0]
    sel_str = f'select pred, ///{chain_id}/{"+".join(resis)}'
    print('PyMOL selection string (copy paste it in PyMOL CLI):')
    print(sel_str)

    # print residues
    print('')
    print('Residue IDs:')
    for i in atom_df.iloc[idx].drop_duplicates('residue_number').sort_values('residue_number')['node_id'].to_list():
        print(i)
