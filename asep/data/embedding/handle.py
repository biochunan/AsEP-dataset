import importlib
import sys
from typing import Callable, Dict, Tuple

import torch
from loguru import logger


class EmbeddingHandler:
    """Class to handle dynamic embedding functions for protein sequences.

    Attributes:
        embedding_func (Callable[[str], torch.Tensor]): The user-defined function for embedding.
    """

    def __init__(
        self,
        embedding_func: Callable[[str], torch.Tensor] = None,
        script_path: str = None,
        function_name: str = None,
    ):
        """Initialize the EmbeddingHandler with a user-defined function.

        Args:
            embedding_func (Callable[[str], torch.Tensor]): A function that takes a string and returns a 2D torch.Tensor.

        Raises:
            ValueError: If the embedding_func does not meet the required criteria.
        """
        if embedding_func is None and (script_path is None or function_name is None):
            raise ValueError(
                "Please provide either an embedding function or a script path and function name."
            )
        if embedding_func is None:
            embedding_func = self.load_function_from_script(script_path, function_name)
        if not self._validate_function(embedding_func):
            raise RuntimeError(
                "Provided function does not meet the required input/output specifications."
            )
        self.embedding_func = embedding_func

    def _validate_function(self, func: Callable[[str], torch.Tensor]) -> bool:
        """
        Validate the user-defined function to ensure it takes a string and returns a 2D torch.Tensor.

        Args:
            func (Callable[[str], torch.Tensor]): The function to validate.

        Returns:
            bool: True if the function is valid, False otherwise.
        """
        dummy_input = "ACDEFGHIKLMNPQRSTVWY"
        # check if it works with dummy input
        try:
            output = func(dummy_input)
        except Exception as e:
            logger.error(
                f"Error when calling the function with dummy input {dummy_input}.\nError info: {e}"
            )
            return False
        # check if the output is a 2D tensor
        try:
            assert isinstance(output, torch.Tensor)
        except AssertionError:
            logger.error(f"Output is not a torch.Tensor. Got {type(output)}")
            return False
        # check if the output is a 2D tensor
        try:
            assert len(output.shape) == 2
        except AssertionError:
            logger.error(f"Output is not a 2D tensor. Got shape {output.shape}")
            return False
        # check the shape of the output
        try:
            assert output.shape[0] == len(dummy_input)
        except AssertionError:
            logger.error(
                f"Output tensor 1st dim does not match the input length. Got shape {output.shape}, which should equal to {len(dummy_input)}"
            )
            return False
        return True  # Simplified for illustration

    def load_function_from_script(
        self, script_path: str, function_name: str
    ) -> Callable:
        """
        Load a function from a given script path.

        Args:
            script_path (str): The path to the user script.
            function_name (str): The name of the function to load.

        Returns:
            Callable: The loaded function.
        """
        # create a module spec from the script path
        spec = importlib.util.spec_from_file_location("user_module", script_path)
        # create a module from the spec
        user_module = importlib.util.module_from_spec(spec)
        # add the module to sys.modules
        sys.modules["user_module"] = user_module
        # execute the module
        spec.loader.exec_module(user_module)
        # get the function from the module
        return getattr(user_module, function_name)

    def embed(self, protein_sequence: str) -> torch.Tensor:
        """
        Embed a protein sequence using the user-defined embedding function.

        Args:
            protein_sequence (str): The protein sequence to embed.

        Returns:
            torch.Tensor: The resulting embedding tensor.
        """
        return self.embedding_func(protein_sequence)


# Example usage:
def example_custom_embedding(protein_sequence: str) -> torch.Tensor:
    # Placeholder for a user-defined embedding function
    return torch.randn(len(protein_sequence), 10)  # Example output shape (L, 10)


if __name__ == "__main__":
    # Initialize with the custom function
    handler = EmbeddingHandler(embedding_func=example_custom_embedding)

    # Example protein sequence
    protein_sequence = "ACDEFGHIKLMNPQRSTVWY"
    embedding_tensor = handler.embed(protein_sequence)
    print(embedding_tensor.shape)
