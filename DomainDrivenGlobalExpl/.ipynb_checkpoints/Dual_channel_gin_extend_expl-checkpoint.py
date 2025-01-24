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
            
            self.num_params = motif_params.size(0)
            self.motif_params = nn.Parameter(motif_params, requires_grad=True)
            self.lookup = lookup
            self.test_lookup = test_lookup

    def motif_to_node_params(self, smiles, num_nodes, batch, device, ignore_unknowns = False):
        
        if not isinstance(smiles, list):
            smiles = [smiles]

        param_list = [None] * num_nodes
        lookup_dict = self.lookup if all(smile in self.lookup for smile in smiles) else self.test_lookup
        nodes_to_motif = [lookup_dict[sm] for sm in smiles]

        batch_offsets = torch.cumsum(
            torch.cat([torch.tensor([0]).to(device), torch.bincount(batch)]), dim=0
        )[:-1]

        for gid, d in enumerate(nodes_to_motif):
            graph_offset = batch_offsets[gid]
            for node_index, (motif_string, motif_index) in d.items():
                index_of_node_in_batch = graph_offset + node_index
                if motif_index is None:
                    if ignore_unknowns:
                        param_list[index_of_node_in_batch] = torch.tensor([0]* self.num_classes, device=device)
                    else:
                        param_list[index_of_node_in_batch] = torch.tensor([1]*self.num_classes, device=device)
                else:
                    param_list[index_of_node_in_batch] = self.motif_params[motif_index].sigmoid()
        #Check the dataset for inconsistencies
        try:
            assert None not in param_list, "Some parameters are not assigned"
        except:
            pdb.set_trace() 
        
        return torch.stack(param_list, dim=0).to(device)
            


    def forward(self, x, edge_index, batch=None, smiles = None,edge_weight = None, ignore_unknowns=False, return_logit=False):
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if edge_weight is not None:
            print("PostHoc support pending")
            exit()
            
        if self.use_ones:
            node_weights = None
            # x.to(edge_index.device)
            
            all_embeddings = ()
            
            for channel in range(self.num_classes):

                #Node Embeddings
                x_channel = self.embedding(x, edge_index, channel)
                #Graph Embedding
                x_channel = global_add_pool(x_channel, batch)
                x_channel = self.classification(x_channel, channel)
                
                all_embeddings = all_embeddings + (x_channel,)

        else:
            node_weights = self.motif_to_node_params(smiles, x.shape[0], batch, x.device, ignore_unknowns)
            # input(node_weights.unique())
            # input(self.num_classes)
            
            node_weights =  node_weights.to(edge_index.device)
            
            all_embeddings = ()
            
            for channel in range(self.num_classes):
            
                # Channel embedding
                x_channel = self.get_graph_representation(x, edge_index, node_weights, channel, batch)

                
                all_embeddings = all_embeddings + (x_channel,)
            
        return torch.cat(all_embeddings, dim=1), node_weights

#         x = F.log_softmax(torch.cat(all_features, dim=1), dim=-1)

#         return x, node_weights
    
    
    def get_graph_representation(self, x, edge_index, node_weights, class_id, batch):
        node_weights =  node_weights.to(edge_index.device)
        x_cls = x * node_weights[:,class_id].unsqueeze(-1)
        x_cls = self.embedding(x_cls, edge_index, class_id)

        # Readout phase: global mean pooling
        x_cls = global_add_pool(x_cls* node_weights[:,class_id].unsqueeze(-1), batch)
        
        # Classification
        return self.classification(x_cls, class_id)
    
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
    
#     def get_masked_graphs(self, loader, motif_idx):
#         '''
#         Checks meaningfulness of each motif importance
#         '''
#         lookup_dict = self.lookup if all(smile in self.lookup for smile in smiles) else self.test_lookup
        
#         new_loader = loader.copy()

#         #Remove the current motif if it exists in the batch
#         for data in new_loader:
#             device = data.device

#             batch_offsets = torch.cumsum(
#                 torch.cat([torch.tensor([0]).to(device), torch.bincount(data.batch)]), dim=0
#             )[:-1]

#             for gid,smile in enumerate(data.smiles):
#                 graph_offset = batch_offsets[gid]

#                 node_to_motif = lookup_dict[smile]
#                 graph_mask = [True] * data.x.shape[0]
#                 for node_index, (motif_string, motif_index) in node_to_motif.items():
#                     if motif_idx == motif_index:
#                         #Exclude node
#                         data.x[graph_offset + node_index] = torch.zeros(data.x.shape[1])

#         return new_loader

 