from typing import Any

import torch
from torch_geometric.data import Data as PygData


# AbAg graph pair data
class PairData(PygData):
    """
    create a PairData object that inherits PyGData object,
    attrs:
    Ab and Ag node features      => x_b and x_g
    Ab and Ag inner graph edges  => edge_index_b and edge_index_g
    Ab-Ag bipartite edges        => edge_index_bg
    """

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index_b":
            return self.x_b.size(0)
        if key == "edge_index_g":
            return self.x_g.size(0)
        if key == "edge_index_bg":
            return torch.tensor([[self.x_b.size(0)], [self.x_g.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
