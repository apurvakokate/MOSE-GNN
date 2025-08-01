import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from collections import defaultdict, deque
import os
from DataLoader import build_graph
from typing import Any, List, Optional, Sequence, Union
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_collate(batch):
    tree_batch = []
    graph_batch = []
    for item in batch:
        tree_batch.append(item)
        graph_batch.append(item.data)
    tree_batch = Batch.from_data_list(tree_batch)
    graph_batch = Batch.from_data_list(graph_batch)

    return (tree_batch, graph_batch)

class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            **kwargs,
        )
        

def sanitize_mol(mol):
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    return mol

def combine_mols(mol1, mol2):
    combo = Chem.CombineMols(mol1, mol2)
    new_mol = Chem.RWMol(combo)
    sanitize_mol(new_mol)
    return new_mol

def motif_is_important(motif):
    mol = get_mol(motif)
    return mol.GetNumHeavyAtoms() > 5

class MAGE:
    def __init__(self, gnn, model, dataset, smiles_set, hidden_channels, output_channels):
        self.gnn = gnn
        self.dataset = dataset
        self.smiles_set = smiles_set
        self.label = 0
        self.motif_id = {}
        self.device = device
        self.hidden_channels = hidden_channels
        self.model = model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.T_mean = Linear(hidden_channels, hidden_channels).to(device)
        self.T_var = Linear(hidden_channels, hidden_channels).to(device)
        self.T_encoder = TGCN(hidden_channels, hidden_channels, output_channels).to(device)
        self.pred_node_topo = Linear(hidden_channels, 2).to(device)
        self.pred_node_label = Linear(hidden_channels, len(smiles_set)).to(device)
        self.linear_topo = Linear(hidden_channels, hidden_channels).to(device)
        self.linear_label = Linear(hidden_channels, hidden_channels).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        trees = [self.get_tree(graph) for graph in tqdm(self.dataset)]
        self.trees = [t for t in trees if t is not None]
        
        self.get_motif_embedding()

    def get_motif_embedding(self):
        # Get the motif embedding from the target GNN model
        print("Getting motif embedding!")
        self.motif_embedding = []
        
        self.model.eval()
        with torch.no_grad():
            for motif in tqdm(self.smiles_set):
                data = build_graph(motif, None, None)
                data.to(self.device)
                batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
                motif_embedding_output = self.model(data.x, data.edge_index, batch)
                motif_embedding = self.model.readout_output
                self.motif_embedding.append(motif_embedding)
                
        self.motif_embedding = torch.stack(self.motif_embedding).squeeze().to(self.device)
        self.pred_node_label = nn.Linear(self.hidden_channels, self.motif_embedding.size(0)).to(self.device)
        
    def get_tree(self, data, test=False):
        node_motif_map = data.nodes_to_motifs

        # Skip graphs where all nodes have motif -1
        if torch.all(node_motif_map == -1):
            return None

        # Convert to flat list and remove -1 entries
        motif_indices = [int(m) for m in node_motif_map if m != -1]
        # Convert to set (removes duplicates)
        motif_indices_set = set(motif_indices)

        # Create PyTorch tensor (sorted for consistency)
        x  = torch.tensor(sorted(motif_indices_set), dtype=torch.long)

        #use data.edge_index to create a new edge index for the motifs
        motif_edge_index = []
        for edge in data.edge_index.t():  # transpose to iterate over edges
            src, dst = edge[0].item(), edge[1].item()
            motif_src, motif_dst = node_motif_map[src].item(), node_motif_map[dst].item()
            # input(f"{motif_src}, {motif_dst},{motif_edge_index}")
            if motif_src == -1 or motif_dst == -1 or motif_src == motif_dst:
                continue
                
            motif_edge_index.append([torch.where(x == motif_src)[0], torch.where(x == motif_dst)[0]])
            motif_edge_index.append([torch.where(x == motif_dst)[0], torch.where(x == motif_src)[0]])
        if len(motif_edge_index) == 0:
            return None #motif_edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            motif_edge_index = torch.tensor(motif_edge_index, dtype=torch.long).t().contiguous()
        if motif_edge_index.numel() == 0:
            print("Warning: Empty motif_edge_index")
        elif motif_edge_index.max() >= len(self.smiles_set):
            raise ValueError(f"Invalid edge_index: max={motif_edge_index.max()} but num_motifs={len(self.smiles_set)}")
        tree_data = Data(x=x, edge_index=motif_edge_index, node_ori_map=node_motif_map, data=data)
        return tree_data

    def encode_tree(self, tree):
        return self.T_encoder(self.motif_embedding[tree.x.view(-1)], tree.edge_index.to(device), batch=tree.batch.to(device), return_embedding=True)

    def encode_graph(self, graph):
        self.model.eval()
        graph = graph.to(device)
        self.model(graph.x, graph.edge_index, graph.batch)
        return self.model.readout_output

    def decode_tree(self, t_data, g_data, z_tree, max_iter):
        t_data, g_data, z_tree = t_data.to(device), g_data.to(device), z_tree.to(device)
        topo_pred = self.pred_node_topo(z_tree)
        label_pred = self.pred_node_label(z_tree)
        topo_loss = self.criterion(topo_pred, t_data.y.to(device))
        label_loss = self.criterion(label_pred, t_data.labels.to(device))
        pred_graph = self.decode_graph(t_data)
        pred_loss = self.mse_loss(pred_graph, g_data.y.to(device))
        emb_loss = self.mse_loss(z_tree, self.encode_graph(g_data))
        acc = (topo_pred.argmax(dim=1) == t_data.y.to(device)).float().mean().item()
        count = t_data.y.size(0)
        return topo_loss, label_loss, pred_loss, emb_loss, acc, count, topo_pred, label_pred

    def decode_graph(self, tree):
        queue = deque()
        for i in range(0, tree.edge_index.shape[1], 2):
            queue.append((tree.edge_index[0, i].item(), tree.edge_index[1, i].item()))
        curr_mol = None
        visited = set()
        while queue:
            u, v = queue.popleft()
            if u in visited and v in visited:
                continue
            motif_u = self.smiles_set[tree.x[u].item()]
            motif_v = self.smiles_set[tree.x[v].item()]
            mol_u = sanitize_mol(get_mol(motif_u))
            mol_v = sanitize_mol(get_mol(motif_v))
            if curr_mol is None:
                curr_mol = mol_u
            curr_mol = combine_mols(curr_mol, mol_v)
            visited.update([u, v])
        smiles = Chem.MolToSmiles(curr_mol)
        data = build_graph(smiles, None, None).to(device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        pred, _ = self.model(data.x, data.edge_index, batch)
        embedding = self.model.readout_output.detach().to(device)
        return embedding

    def sample_tree(self, z_tree, max_iter, test=False):
        curr_x = z_tree.clone().to(device)
        outputs = []
        for _ in range(max_iter):
            topo_logits = self.pred_node_topo(curr_x)
            label_logits = self.pred_node_label(curr_x)
            topo_sample = torch.multinomial(F.softmax(topo_logits, dim=-1), 1)
            label_sample = torch.multinomial(F.softmax(label_logits, dim=-1), 1)
            outputs.append((topo_sample.detach().to(device), label_sample.detach().to(device)))
            curr_x = torch.cat((curr_x, F.one_hot(label_sample, num_classes=len(self.smiles_set)).float().to(device)), dim=-1)
        return outputs

    def get_motif_mask(self):
        motif_masks = torch.zeros(len(self.smiles_set), device=device)
        for idx, motif in enumerate(self.smiles_set):
            if motif_is_important(motif):
                motif_masks[idx] = 1.0
        self.motif_masks = motif_masks

    def rsample(self, embedding, T_mean, T_var):
        embedding = embedding.to(device)
        mu, var = T_mean(embedding), torch.exp(0.5 * T_var(embedding))
        eps = torch.randn_like(var).to(device)
        z = mu + eps * var
        kl_div = -0.5 * torch.sum(1 + torch.log(var.pow(2)) - mu.pow(2) - var.pow(2))
        return z, kl_div

    def train(self, epochs, batch_size, lr, max_iter, path_dict, t_encoder_path):
        self.get_motif_mask()
        self.T_encoder.load_state_dict(torch.load(t_encoder_path, map_location=device))
        optimizer = torch.optim.Adam(
            list(self.T_encoder.parameters()) +
            list(self.pred_node_topo.parameters()) +
            list(self.pred_node_label.parameters()) +
            list(self.linear_topo.parameters()) +
            list(self.linear_label.parameters()) +
            list(self.T_mean.parameters()) +
            list(self.T_var.parameters()),
            lr=lr
        )
        train_loader = DataLoader(self.trees, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            for t_data, g_data in tqdm(train_loader):
                t_data, g_data = t_data.to(device), g_data.to(device)
                tree_emb = self.encode_tree(t_data)
                z_tree, kl_tree = self.rsample(tree_emb, self.T_mean, self.T_var)
                topo_loss, label_loss, pred_loss, emb_loss, acc, count, _, _ = self.decode_tree(t_data, g_data, z_tree, max_iter)
                loss = kl_tree + 10 * pred_loss + emb_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch}: Loss = {total_loss}')
            if total_loss < best_loss:
                best_loss = total_loss
                self.save(path_dict)

    def save(self, path_dict):
        torch.save(self.T_encoder.state_dict(), path_dict['T_encoder'])
        torch.save(self.T_mean.state_dict(), path_dict['T_mean'])
        torch.save(self.T_var.state_dict(), path_dict['T_var'])
        torch.save(self.pred_node_topo.state_dict(), path_dict['pred_node_topo'])
        torch.save(self.pred_node_label.state_dict(), path_dict['pred_node_label'])
        torch.save(self.linear_topo.state_dict(), path_dict['linear_topo'])
        torch.save(self.linear_label.state_dict(), path_dict['linear_label'])

class TGCN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(TGCN, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, return_embedding=False):
        x, edge_index = x.to(device), edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        if batch is not None:
            x = global_mean_pool(x, batch.to(device))
        return x if return_embedding else self.lin(x)


