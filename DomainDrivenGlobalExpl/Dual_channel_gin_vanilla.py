import torch.nn as nn
from torch_geometric.nn import global_add_pool, TopKPooling
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential, Dropout
import torch
from torch_scatter import scatter_add
from Utils_Train import evaluate_model
from Utils_model import create_conv_layers 


class GNNModel(nn.Module): 
    def __init__(self,input_dim, output_dim, hidden_channels, num_layers, layer_type, use_explainer=False,
                motif_params=None, lookup=None, test_lookup=None, task_type = 'BinaryClass'):
        super().__init__()
        num_mp_layers  = num_layers
        hidden         = hidden_channels
        
        self.num_classes = output_dim
        self.task_type = task_type
        
        # Create dictionaries to hold the convolutional and linear layers for each class
        self.convs = nn.ModuleDict()
        self.lin1 = nn.ModuleDict()
        self.lin2 = nn.ModuleDict()

        for i in range(self.num_classes):
            self.convs[str(i)] = create_conv_layers(input_dim, hidden_channels, num_layers, layer_type)
            self.lin1[str(i)] = Linear(hidden_channels, hidden_channels)
            self.lin2[str(i)] = Linear(hidden_channels, 1)
        
        self.use_ones = True
            
    def forward(self, x, edge_index, batch=None, smiles = None,edge_weight = None, ignore_unknowns=False):
        
        if edge_weight is not None:
            print("PostHoc support pending")
            exit()
            
            x.to(edge_index.device)
            
        all_features = ()
            
        for channel in range(self.num_classes):
            
            # Class 0 embedding
            x_channel = self.embedding(x, edge_index, channel)
            #Graph Embedding
            x_channel = global_add_pool(x_channel, batch)
            x_channel = self.classification(x_channel, channel)

            # Class 1 embedding    
            all_features = all_features + (x_channel,)
            
        # if self.task_type == 'Regression':
            # return torch.cat(all_features, dim=1), None

#         x = F.log_softmax(torch.cat(all_features, dim=1), dim=-1)

#         return x, None
        return torch.cat(all_features, dim=1), None
            
    
    def embedding(self, x, edge_index, class_id):
        # x = x.to(edge_index.device)
        for conv in self.convs[str(class_id)]:
            conv = conv.to(edge_index.device)
            x = conv(x, edge_index)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = F.relu(x)
        return x
    
    def classification(self, x, class_id):
        # self.lin1[str(class_id] = self.lin1[class_id].to(x.device)
        x = F.relu(self.lin1[str(class_id)](x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2[str(class_id)](x)
        return x
    