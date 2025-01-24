import torch
from torch_geometric.nn import GINConv, GCNConv, GATConv
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d as BN

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

    return convs