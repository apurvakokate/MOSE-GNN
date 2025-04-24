import torch
from torch import Tensor
from torch_geometric.nn import GINConv, GCNConv, GATConv
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d as BN
from torch_geometric.nn import GINConv as BaseGINConv
from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor

def create_conv_layers(in_channels, hidden_channels, num_mp_layers, layer_type):
    convs = torch.nn.ModuleList()

    # Handle the first layer
    if layer_type == 'GINConv':
        convs.append(
            GINConv(
                Sequential(
                    Linear(in_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    BN(hidden_channels),
                ), train_eps=True))
    elif layer_type == 'GCNConv':
        convs.append(GCNConv(in_channels, hidden_channels))
    elif layer_type == 'GATConv':
        convs.append(GATConv(in_channels, hidden_channels))
    elif layer_type == 'WeightedGIN':
        convs.append(
            WeightedGINConv(
                Sequential(
                        Linear(in_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels)
                    ), train_eps=True))
        

    # Handle the remaining layers
    for i in range(num_mp_layers - 1):
        if layer_type == 'GINConv':
            convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels)
                    ), train_eps=True))
        elif layer_type == 'GCNConv':
            convs.append(GCNConv(hidden_channels, hidden_channels))
        elif layer_type == 'GATConv':
            convs.append(GATConv(hidden_channels, hidden_channels))
        elif layer_type == 'WeightedGIN':
            convs.append(
                WeightedGINConv(
                    Sequential(
                            Linear(hidden_channels, hidden_channels),
                            ReLU(),
                            Linear(hidden_channels, hidden_channels),
                            ReLU(),
                            BN(hidden_channels)
                        ), train_eps=True))

    return convs

class WeightedGINConv(BaseGINConv):
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