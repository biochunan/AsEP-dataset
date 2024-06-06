# basic
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from Bio import AlignIO, SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord


# align seq using ClustalOmega
def run_align_clustalomega(clustal_omega_executable: str,
                           seq1: str = None, seq2: str = None,
                           seqs: List[str] = None) -> List[SeqRecord]:
    """

    Args:
        seq1: sequence of a chain e.g. seqres sequence
        seq2: sequence of a chain e.g. atmseq sequence
        or you can provide a list of strings using seqs
        seqs: e.g. ["seq1", "seq2", ...]
        clustal_omega_executable: (str) path to clustal omega executable
            e.g. "/usr/local/bin/clustal-omega"
    Returns:
        aln_seq_records: (List)
    """
    # assert input
    if seqs is None and (seq1 is None or seq2 is None):
        raise NotImplemented(f"Provide either List of seqs as `seqs` OR a pair of seqs as `seq1` and `seq2`.")

    # generate seq_recs
    seq_rec = [None]
    if seqs:
        seq_rec = [SeqRecord(id=f"seq{i + 1}", seq=Seq(seqs[i]), description="")
                   for i in range(len(seqs))]
    elif seq1 is not None and seq2 is not None:
        seq_rec = [SeqRecord(id=f"seq{1}", seq=Seq(seq1), description=""),
                   SeqRecord(id=f"seq{2}", seq=Seq(seq2), description="")]

    with tempfile.TemporaryDirectory() as tmpdir:
        # executable
        cmd = clustal_omega_executable

        # create input seq fasta file and output file for clustal-omega
        in_file = os.path.join(tmpdir, "seq.fasta")
        out_file = os.path.join(tmpdir, f"aln.fasta")
        with open(in_file, "w") as f:
            SeqIO.write(seq_rec, f, "fasta")
        # create Clustal-Omega commands
        clustalomega_cline = ClustalOmegaCommandline(cmd=cmd, infile=in_file, outfile=out_file, verbose=True, auto=True)

        # run Clustal-Omega
        stdout, stderr = clustalomega_cline()

        # read aln
        aln_seq_records = []
        with open(out_file, "r") as f:
            for record in AlignIO.read(f, "fasta"):
                aln_seq_records.append(record)

        return aln_seq_records


def gen_seqres_to_atmseq_mask(seqres: str,
                              atmseq: str,
                              clustal_omega_executable: Union[Path, str]) -> List[int]:
    aln = run_align_clustalomega(clustal_omega_executable=clustal_omega_executable,
                                 seq1=seqres,
                                 seq2=atmseq)
    # assert no dash in seqres
    try:
        assert '-' not in str(aln[0].seq)
    except AssertionError:
        print(f'seqres: {seqres}')
        print(f'atmseq: {atmseq}')
        print(f'aln[0].seq: {aln[0].seq}')
        print(f'aln[1].seq: {aln[1].seq}')
        raise AssertionError('seqres contains dash, please check your input seqres and atmseq')

    aln1 = str(aln[1].seq)  # => this is the atmseq in aln, may contain "-"
    seqres_to_atmseq_mask=[1 if i != '-' else 0 for i in aln1]  # 1 => in atmseq; 0 => not in atmseq

    return seqres_to_atmseq_mask


def gen_mask(
        seqres: str,                                 # seqres sequence
        atmseq: str,                                 # atmseq sequence
        df_in: pd.DataFrame,                         # chain df
        clustal_omega_executable: Union[Path, str],  # clustal omega executable
        col: Optional[str]=None,                     # boolean column name e.g. "is_epitope"
) -> Dict:  # sourcery skip: extract-method, move-assign, remove-redundant-if
    """
    Generate mask for `seqres2atmseq` and `cdr_mask`, both have the same length as `seqres`

    Args:
        seqres (str): SEQRES sequence of a chain e.g. heavy chain
        atmseq (str): ATMSEQ sequence of a chain e.g. heavy chain
        df_in (pd.DataFrame): structure DataFrame of the input chain
        col (str): col name of a boolean column e.g. "is_cdr"

    Returns:
        mask:
            keys: "seqres2atmseq", "cdr_mask"
            vals: lists of binary elements, of the same length as `len(seqres)`
        e.g. 1fgv_0 heavy chain
        for simplicity, the values are shown as strings, in reality they are lists of integers i.e. list(map(int, mask["seqres2atmseq"]))
        mask => {
            "seqres2atmseq": "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110000111111111111111111",
            "cdr_mask": "0000000000000000000000000111111111100000000000000111111111100000000000000000000000000000000000000011110000111111100000000000"
        }
    """
    df = df_in.copy()

    if col:
        assert df is not None, "Error: if col is provided, df must be provided"
        assert col in df, f"Error: df must contain '{col}' column"

    # run clustal omega seq alignment between `seqres` and `atmseq`
    aln = run_align_clustalomega(
        clustal_omega_executable=clustal_omega_executable,
        seq1=seqres,
        seq2=atmseq
    )
    assert "-" not in str(aln[0].seq), "Error: seqres contains dash"

    aln1 = str(aln[1].seq)  # atmseq in aln may contain "-"
    n = len(aln1)  # aln cols should equal to len(seqres)

    # 1. add seqres2atmseq mask
    mask = dict(
        seqres2atmseq=[1 if i != "-" else 0 for i in aln1]  # 1 => in atmseq; 0 => not in atmseq
    )

    # 2. add cdr_mask
    if col == "is_cdr":
        mask["cdr_mask"] = cdr_mask = []
        j = 0  # j => atmseq index

        # ensure node_id to start from 0
        min_node_id = df.node_id.min()
        if min_node_id > 0:
            df[df.columns[0]] = df["node_id"].astype(int).values - min_node_id
        # convert to dict: node_id => is_cdr (binary)
        d = df[["node_id", col]].drop_duplicates("node_id").set_index("node_id")[col].to_dict()
        '''
        the logic here is
        each position in aln1 has two indices:
        1. atmseq index, this is node_id
        2. aln1   index, length >= len(atmseq), (this is > when "-" is in aln1)
        '''
        # iterate over aln1
        for i in range(n):  # i => aln col index
            c = aln1[i]
            if c != "-":  # a match between atmseq and seqres
                if d.get(j, False):
                    cdr_mask.append(1)  # => this match is a CDR residue
                else:
                    cdr_mask.append(0)  # => this match is a FRAMEWORK residue
                j += 1
            elif c == "-":  # a mismatch i.e. missing residue in ATOM section
                # j 不变
                # append 0 regardless of the residue identity being CDR or not (i.e. missing residue)
                # e.g. missing CDR residue; missing FRAMEWORK residue;
                cdr_mask.append(0)

    return mask

