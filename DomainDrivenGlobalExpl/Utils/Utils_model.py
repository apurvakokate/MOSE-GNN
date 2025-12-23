import torch
from torch import Tensor
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, PNAConv
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d as BN
from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor
from torch_geometric.utils import degree

def compute_deg(loader, num_nodes=None):
    """Compute the degree histogram from a PyTorch Geometric DataLoader."""
    deg_hist = torch.zeros(0, dtype=torch.long)

    for data in loader:
        if not hasattr(data, 'edge_index'):
            continue
        row = data.edge_index[0]
        deg_batch = degree(row, num_nodes=data.num_nodes, dtype=torch.long)
        deg_hist = torch.cat([deg_hist, deg_batch])

    deg = torch.bincount(deg_hist)
    return deg

def create_conv_layers(in_channels, hidden_channels, num_mp_layers, layer_type, deg, weightedEdge = False):
    convs = torch.nn.ModuleList()

    # First layer
    if layer_type == 'GIN':
        if weightedEdge:
            convs.append(
                WeightedGINConv(
                    Sequential(
                            Linear(in_channels, hidden_channels),
                            ReLU(),
                            Linear(hidden_channels, hidden_channels),
                            ReLU(),
                            BN(hidden_channels)
                        ), train_eps=True))
        else:
            convs.append(
                GINConv(
                    Sequential(
                        Linear(in_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ), train_eps=True))
    elif layer_type == 'GCN':
        convs.append(GCNConv(in_channels, hidden_channels))
    elif layer_type == 'GAT':
        convs.append(GATConv(in_channels, hidden_channels))
    elif layer_type == 'SAGE':
        convs.append(SAGEConv(in_channels, hidden_channels))
    elif layer_type == 'PNA':
        convs.append(PNAConv(in_channels, hidden_channels, aggregators=['mean', 'min', 'max', 'std'], scalers=['identity', 'amplification', 'attenuation'], deg= deg))
    else:
        raise Exception("Invalid layer type") 
        

    # Handle the remaining layers
    for i in range(num_mp_layers - 1):
        if layer_type == 'GIN':
            if weightedEdge:
                convs.append(
                    WeightedGINConv(
                        Sequential(
                                Linear(hidden_channels, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels),
                                ReLU(),
                                BN(hidden_channels)
                            ), train_eps=True))
            else:
                convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden_channels, hidden_channels),
                            ReLU(),
                            Linear(hidden_channels, hidden_channels),
                            ReLU(),
                            BN(hidden_channels),
                        ), train_eps=True))
            
        elif layer_type == 'GCN':
            convs.append(GCNConv(hidden_channels, hidden_channels))
        elif layer_type == 'GAT':
            convs.append(GATConv(hidden_channels, hidden_channels))
        elif layer_type == 'SAGE':
            convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif layer_type == 'PNA':
            convs.append(PNAConv(in_channels, hidden_channels, aggregators=['mean', 'min', 'max', 'std'], scalers=['identity', 'amplification', 'attenuation'], deg= deg))

    return convs

class WeightedGINConv(GINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        else:
            return x_j