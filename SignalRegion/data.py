import torch
from torch_geometric.data import Data

#edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
print(data)
print(data.keys)
print(data.num_nodes)
print(data.num_edges)
print(data.has_isolated_nodes())
print(data.has_self_loops())
print(data.is_directed())

