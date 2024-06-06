"""
Embed a protein sequence with BLOSUM62 matrix
"""

from typing import Dict, Tuple

import torch
from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")
# BLOSUM62.alphabet  # => 'ARNDCQEGHILKMFPSTWYVBZX*'


def create_blosum62_tensor() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Creates a tensor from the BLOSUM62 substitution matrix using the new Biopython module."""
    blosum62 = substitution_matrices.load("BLOSUM62")
    amino_acids = blosum62.alphabet
    aa_index = {aa: idx for idx, aa in enumerate(amino_acids)}
    matrix_size = len(amino_acids)
    blosum62_tensor = torch.zeros((matrix_size, matrix_size), dtype=torch.float32)
    for i, aa1 in enumerate(amino_acids):
        for j, aa2 in enumerate(amino_acids):
            blosum62_tensor[i, j] = blosum62[aa1, aa2]
    return blosum62_tensor, aa_index


def protein_to_embedding_torch(
    protein_sequence: str, blosum62_tensor: torch.Tensor, aa_index: dict
) -> torch.Tensor:
    """
    Convert a protein sequence to an embedding using a pre-defined BLOSUM62 tensor.

    Args:
        protein_sequence (str): The protein sequence.
        blosum62_tensor (torch.Tensor): The BLOSUM62 tensor.
        aa_index (dict): Mapping from amino acids to indices in the tensor.

    Returns:
        torch.Tensor: A tensor representing the BLOSUM62-based embeddings of the sequence.
    """
    indices = torch.tensor(
        [aa_index.get(aa, -1) for aa in protein_sequence if aa in aa_index],
        dtype=torch.long,
    )
    return blosum62_tensor[indices]


# entrypoint
def embed_blosum62(protein_sequence: str) -> torch.Tensor:
    blosum62_tensor, aa_index = create_blosum62_tensor()
    return protein_to_embedding_torch(
        protein_sequence=protein_sequence,
        blosum62_tensor=blosum62_tensor,
        aa_index=aa_index)
