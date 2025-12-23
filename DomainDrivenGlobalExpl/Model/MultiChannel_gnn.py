import torch.nn as nn
from torch_geometric.nn import global_add_pool, TopKPooling
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential, Dropout
import torch
from torch_scatter import scatter_add
from Utils_Train import evaluate_model
from Utils_model import create_conv_layers 
import pdb

class GNNModel(nn.Module): 
    """Multi-label GNN with per-class conv/MLP stacks and optional motif explainer.

    Logic preserved from the original implementation.
    - Each class `c` has its own conv stack `convs[str(c)]` and MLP `lin1/lin2`.
    - `motif_params` is expected to have shape [num_motifs, num_classes].
    - When `use_explainer=False`, node weights are effectively all-ones.
    - Outputs are concatenated across classes to shape [num_graphs, num_classes].
    """
    def __init__(self,input_dim, output_dim, hidden_channels, num_layers, layer_type, use_explainer=False,
                motif_params=None, lookup=None, test_lookup=None, task_type = 'MultiLabel'):
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
        
        if not use_explainer:
            print("No Explainer parameters will be used. Assuming all node weights are 1")
            self.use_ones = True
        else:
            self.use_ones = False
            self.motif_params = nn.Parameter(motif_params, requires_grad=True)
            self.lookup = lookup
            self.test_lookup = test_lookup

    def motif_to_node_params(self, node_to_motifs, num_nodes, device, ignore_unknowns = False):
        
        if ignore_unknowns:
            param_tensor = torch.full((node_to_motifs.shape[0], self.num_classes), 0.0, device=device)
        else:
            param_tensor = torch.full((node_to_motifs.shape[0], self.num_classes), 1.0, device=device)
            
        for index_of_node_in_batch, motif_index in enumerate(node_to_motifs):
            if motif_index != -1:
                param_tensor[index_of_node_in_batch] = self.motif_params[motif_index].sigmoid()
           
        return param_tensor


    def forward(self, x, edge_index, batch=None, node_to_motifs = None,edge_weight = None, ignore_unknowns=False, return_logit=False):
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if edge_weight is not None:
            print("PostHoc support pending")
            exit()
            
        if self.use_ones:
            node_weights = None
            # x.to(edge_index.device)
            
            all_output = ()
            self.all_readout_output = []
            
            for channel in range(self.num_classes):

                #Node Embeddings
                x_channel = self.embedding(x, edge_index, channel)
                #Graph Embedding
                x_channel = global_add_pool(x_channel, batch)
                self.all_readout_output.append(x_channel)
                x_channel = self.classification(x_channel, channel)
                
                all_output = all_output + (x_channel,)
                
            self.readout_output = torch.stack(self.all_readout_output, dim=0)
            pdb.set_trace()

        else:
            node_weights = self.motif_to_node_params(node_to_motifs, x.shape[0], x.device, ignore_unknowns)
            
            node_weights =  node_weights.to(edge_index.device)
            
            all_output = ()
            self.all_readout_output = []
            
            
            for channel in range(self.num_classes):
            
                # Channel embedding
                x_channel = self.get_graph_representation(x, edge_index, node_weights, channel, batch)

                
                all_output = all_output + (x_channel,)
                
            self.readout_output = torch.stack(self.all_readout_output, dim=0)
            
        return torch.cat(all_output, dim=1), node_weights
    
    
    def get_graph_representation(self, x, edge_index, node_weights, class_id, batch):
        node_weights =  node_weights.to(edge_index.device)
        x_cls = x * node_weights[:,class_id].unsqueeze(-1)
        x_cls = self.embedding(x_cls, edge_index, class_id)

        # Readout phase: global mean pooling
        x_cls = global_add_pool(x_cls* node_weights[:,class_id].unsqueeze(-1), batch)
        
        self.all_readout_output.append(x_cls)
        
        # Classification
        return self.classification(x_cls, class_id)
    
    def embedding(self, x, edge_index, class_id):
        # x = x.to(edge_index.device)
        for conv in self.convs[str(class_id)]:
            conv = conv.to(edge_index.device)
            # pdb.set_trace()
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