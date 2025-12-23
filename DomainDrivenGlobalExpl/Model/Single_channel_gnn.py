import torch.nn as nn
from torch_geometric.nn import global_add_pool, TopKPooling
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential, Dropout
from torch_geometric.nn.conv import GINConv
import torch
from Utils.Utils_Train import evaluate_model
from Utils.Utils_model import create_conv_layers 
import pdb
class GNNModel(nn.Module): 
    """Graph Neural Network with optional motif-based explainer weights.

    Parameters
    ----------
    input_dim : int
        Node feature dimensionality.
    output_dim : int
        Number of classes (for classification) or output units.
    hidden_channels : int
        Hidden dimensionality used across convolution and MLP layers.
    num_layers : int
        Number of message-passing layers.
    layer_type : str
        Convolution layer type understood by `create_conv_layers`.
    use_explainer : bool, optional (default: False)
        If True, multiply node features by learnable per-motif weights before
        message passing and use these weights again during readout.
    motif_params : Optional[Tensor]
        Shape: [num_motifs, 1] or [num_motifs]. Values will be learned
        (sigmoid-squashed) as node weights for nodes that map to each motif.
    lookup : Optional[dict]
        (Reserved) Mapping used elsewhere; stored for completeness.
    test_lookup : Optional[dict]
        (Reserved) Mapping used for test-time motifs; stored for completeness.
    task_type : str, optional (default: "BinaryClass")
        Task identifier; currently unused but kept for compatibility.
    deg : Optional[Tensor]
        Degree histogram for PNA or similar conv types.
    """
    def __init__(self,input_dim, output_dim, hidden_channels, num_layers, layer_type, use_explainer=False,
                motif_params=None, lookup=None, task_type= 'BinaryClass', test_lookup=None, unk_importance = 1.0,
                 use_annealing = False, gumbel_tau = 1.0, use_gumbel_softmax = False, use_stl = False, use_zero_weight = False, learn_unknown = False):
        super().__init__()
        num_mp_layers  = num_layers
        hidden         = hidden_channels
        
        self.num_classes = output_dim
        self.task_type = task_type
        
        self.convs = create_conv_layers(input_dim, hidden_channels*2, num_layers, layer_type)
        self.lin1 = Linear(hidden_channels*2, hidden_channels*2)
        self.lin2 = Linear(hidden_channels*2, self.num_classes)
        
        if not use_explainer:
            print("No Explainer parameters will be used. Assuming all node weights are 1")
            self.use_ones = True
        else:
            self.use_ones = False
            self.motif_params = nn.Parameter(motif_params, requires_grad=True)
            self.learnable_unk = learn_unknown
            if learn_unknown:
                
                # one shared learnable scalar for all unknown motifs
                self.unk_param = torch.nn.Parameter(torch.tensor(unk_importance))
            self.lookup = lookup
            self.test_lookup = test_lookup
            
            # Temperature is a buffer so you can anneal it without breaking state_dict.
            self.register_buffer("gumbel_tau", torch.tensor(float(gumbel_tau)))
            self.unk_importance = unk_importance
            self.use_gumbel_softmax = use_gumbel_softmax
            self.use_stl = use_stl
            self.use_annealing = use_annealing
            self.use_zero_weight = use_zero_weight
        
    # Optional helper if you want to anneal temperature externally each epoch.
    @torch.no_grad()
    def anneal_gumbel_tau(self, factor=0.95, min_tau=0.3):
        self.gumbel_tau.mul_(factor)
        if self.gumbel_tau.item() < min_tau:
            self.gumbel_tau.fill_(min_tau)

    @staticmethod
    def _gumbel_binary_from_logits(logits, tau, hard):
        """
        Binary Gumbel-Softmax via 2-class trick:
        Returns y in [0,1] with shape (..., 1), representing the 'ON' gate.
        If hard=True, forward is {0,1} with ST gradient.
        """
        # logits: (..., 1) or (...,)  -> make it (...,)
        logits = logits.squeeze(-1)
        # Build 2-class logits: class0=0 (OFF), class1=logits (ON)
        two_class = torch.stack([torch.zeros_like(logits), logits], dim=-1)  # (..., 2)
        # ST when hard=True
        y2 = F.gumbel_softmax(two_class, tau=tau, hard=hard, dim=-1)         # (..., 2)
        on = y2[..., 1].unsqueeze(-1)                                        # (..., 1)
        return on
    
    @staticmethod
    def _ste_sigmoid(logits, tau=1.0, hard=True, thresh=0.5):
        # Soft value used for gradients
        y_soft = torch.sigmoid(logits / tau)

        if not hard:
            return y_soft  # purely soft

        # Hard forward (0/1), but gradients = dy_soft/dlogits
        y_hard = (y_soft > thresh).float()
        return y_hard + (y_soft - y_soft.detach())  # straight-through trick

            
    def motif_to_node_params(self, node_to_motifs, num_nodes, device, ignore_unknowns = False, masked_motif=None):
        # torch.set_printoptions(threshold=10_000)
        # input(node_to_motifs)
        if ignore_unknowns:
            param_tensor = torch.full((node_to_motifs.shape[0], 1), 0.0, device=device)
        else:
            if self.learnable_unk:
                unk_val = self.unk_param.sigmoid()  # learnable scalar
                param_tensor = unk_val.expand(node_to_motifs.shape[0], 1).clone()
            else:
                unk_val = self.unk_importance  # fixed scalar
                param_tensor = torch.full((node_to_motifs.shape[0], 1), unk_val, device=device)

        for index_of_node_in_batch, motif_index in enumerate(node_to_motifs):
            if motif_index == masked_motif:
                param_tensor[index_of_node_in_batch] = 0.0
            elif motif_index != -1:
                motif_logit = self.motif_params[motif_index]
                if self.use_stl and self.use_gumbel_softmax and self.training:
                    param_tensor[index_of_node_in_batch] = self._gumbel_binary_from_logits(motif_logit, tau=self.gumbel_tau, hard=True)
                elif self.use_gumbel_softmax and self.training:
                    param_tensor[index_of_node_in_batch] = self._gumbel_binary_from_logits(motif_logit, tau=self.gumbel_tau, hard=False)
                elif self.use_stl:
                    if self.training:
                        param_tensor[index_of_node_in_batch] = self._ste_sigmoid(motif_logit, self.gumbel_tau) 
                    else:
                        param_tensor[index_of_node_in_batch] = (self.motif_params[motif_index].sigmoid() > 0.5).float()
                elif self.use_annealing:
                    param_tensor[index_of_node_in_batch] = self._ste_sigmoid(motif_logit, self.gumbel_tau, hard= False) 
                    # if self.training:
                    #     param_tensor[index_of_node_in_batch] = self._ste_sigmoid(motif_logit, self.gumbel_tau, hard= False) 
                    # else:
                    #     param_tensor[index_of_node_in_batch] = (self.motif_params[motif_index].sigmoid() > 0.5).float()
                else:
                    param_tensor[index_of_node_in_batch] = self.motif_params[motif_index].sigmoid()
            
           
        return param_tensor
            
    def forward(self, x, edge_index,
        batch=None,
        node_to_motifs=None,
        edge_weight=None,
        ignore_unknowns=False,
        return_logit=False,
        masked_motif=None,
    ):
        # batch default
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # not supported yet
        if edge_weight is not None:
            raise NotImplementedError("PostHoc support pending")

        if self.use_ones:
            # Optional node masking when masked_motif is provided
            node_weights = None
            if masked_motif is not None and node_to_motifs is not None:
                # [N] on same device/dtype as x, then make it [N,1] for feature-wise scaling
                node_weights = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
                if not isinstance(node_to_motifs, torch.Tensor):
                    node_to_motifs = torch.as_tensor(node_to_motifs, device=x.device)
                mask = (node_to_motifs.to(x.device) == masked_motif)
                node_weights = node_weights.masked_fill(mask, 0.0).unsqueeze(-1)  # [N,1]

            # Node Embeddings
            if node_weights is not None:
                x = x * node_weights
            x = self.embedding(x, edge_index)

            # Graph Embedding
            if node_weights is not None:
                x = x * node_weights
            x = global_add_pool(x, batch)

            self.readout_output = x  # preserve for downstream use
            x = self.classification(x)

        else:
            # Map motif-level params to per-node weights; ensure shape/device/dtype are compatible
            node_weights = self.motif_to_node_params(
                node_to_motifs,
                x.shape[0],
                x.device,
                ignore_unknowns,
                masked_motif=masked_motif,
            )
            node_weights = node_weights.to(device=x.device, dtype=x.dtype)  # e.g., [N,1]

            # Class 0 embedding / general graph representation path
            x = self.get_graph_representation(x, edge_index, node_weights, batch)

        # If you plan to use return_logit later, handle it here (left as-is).
        return x, node_weights
    
    
    def get_graph_representation(self, x, edge_index, node_weights, batch):
        node_weights =  node_weights.to(edge_index.device)
        x_cls = x * node_weights
        x_cls = self.embedding(x_cls, edge_index)

        # Readout phase: global mean pooling
        x_cls = global_add_pool(x_cls* node_weights, batch)
        
        self.readout_output = x_cls
        
        # Classification
        return self.classification(x_cls)
    
    def embedding(self, x, edge_index):
        for conv in self.convs:
            conv = conv.to(edge_index.device)
            x = conv(x, edge_index)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = F.relu(x)
        return x
    
    def classification(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x