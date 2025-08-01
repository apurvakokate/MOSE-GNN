# Please write an official comment and the code for the Mage class here. The class is taking a pretrained GNN, and the dataset that used to pretrain the GNN. The Mage class is responsible for create a graph encoder, a tree encoder, a tree decoder, and a graph decoder to generate model level explanation for the target GNN.
# The class should have the following methods:
# - encode_graph: This method should take a graph as input and return the encoded graph.
# - encode_tree: This method should take a tree as input and return the encoded tree.
# - decode_tree: This method should take an encoded tree as input and return the decoded tree.
# - decode_graph: This method should take an encoded graph as input and return the decoded graph.
# - explain: This method should sample explanations for the target GNN.

# Start coding here
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from DataLoader import build_graph
from torch_geometric.data import Data, Batch
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdmolops
from sklearn.model_selection import train_test_split
from typing import Any, List, Optional, Sequence, Union
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
import pdb
from torch_geometric.nn import global_mean_pool
import os
import pdb
from motif_filter import motif_filter

class MAGE:
    def __init__(self, gnn, model, dataset, whole_dataset,smiles_set, hidden_channels, output_channels, device):
        self.gnn = gnn
        self.dataset = dataset
        self.whole_dataset= whole_dataset
        self.smiles_set = smiles_set
        self.label = 0 # self.label = label for multi_label
        # self.motif_id = {}
        # self.id_motif = {}
        self.device = device
        self.hidden_channels = hidden_channels
        self.model = model.to(self.device) # Pretrained target GNN model as the graph encoder
        for param in self.model.parameters(): # Freeze the parameter of self.model
            param.requires_grad = False
            
        self.T_mean = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        self.T_var = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        self.T_encoder = TGCN(hidden_channels, hidden_channels, output_channels).to(self.device)
        self.pred_node_topo = nn.Linear(hidden_channels, 2).to(self.device)
        
        self.linear_topo = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        self.linear_label = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        self.get_motif_embedding()

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
        # Create x for the tree, each node is a motif, the node feature is the motif id
        x  = torch.tensor(sorted(motif_indices_set), dtype=torch.long)

        #use data.edge_index to create a new edge index for the motifs
        motif_edge_index = []
        for edge in data.edge_index.t():  
            src, dst = edge[0].item(), edge[1].item()
            
            motif_src, motif_dst = node_motif_map[src].item(), node_motif_map[dst].item()
            
            if motif_src == -1 or motif_dst == -1 or motif_src == motif_dst:
                continue
                
            set1 = torch.where(x == motif_src)
            set2 = torch.where(x == motif_dst)

            for node1 in set1:
                for node2 in set2:
                    motif_edge_index.append([node1, node2])
                    motif_edge_index.append([node2, node1])

        if len(motif_edge_index) == 0:
            motif_edge_index = torch.tensor([[], []], dtype=torch.long) # None to fix
        else:
            motif_edge_index = torch.tensor(motif_edge_index, dtype=torch.long).t().contiguous()
            
        # Verifying edge index    
        if motif_edge_index.numel() == 0:
            print("Warning: Empty motif_edge_index")
        elif motif_edge_index.max() >= len(self.smiles_set):
            raise ValueError(f"Invalid edge_index: max={motif_edge_index.max()} but num_motifs={len(self.smiles_set)}")
        else:
            print(motif_edge_index.numel())
            
        tree_data = Data(x=x, edge_index=motif_edge_index, node_ori_map=node_motif_map, data=data)
        return tree_data
    
    def get_motif_embedding(self):
        trees = [self.get_tree(graph) for graph in tqdm(self.dataset)]
        self.trees = [t for t in trees if t is not None]

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


    def get_motif_mask(self, heter_path):
        mask = torch.zeros((1, len(self.smiles_set)), dtype=torch.long).to(self.device)
        mask_pred = torch.zeros((1, len(self.smiles_set)), dtype=torch.float).to(self.device)
        
        for i in range(self.motif_embedding.size(0)):
            emb = self.motif_embedding[i].unsqueeze(0)
            pred = self.model.classification(emb)
            
            # pred = F.softmax(pred, dim=1)
            pred = F.sigmoid(pred)
            
            #if pred[0, self.label] > 0.99:
            if pred[0, 0] > 0.5:
                mask[0][i] = 1
        self.first_node_mask = mask.to(self.device)
        # if os.path.exists("checkpoints/motif_selection/"+'BBBP'+"_motif_"+str(self.label)+".pt"):
        #     selected_motif = torch.load("checkpoints/motif_selection/"+'BBBP'+"_motif_"+str(self.label)+".pt")
        #     label_0_score = torch.load(f"checkpoints/motif_selection/"+'BBBP'+"_label_0_scores.pt")
        #     label_1_score = torch.load(f"checkpoints/motif_selection/"+'BBBP'+"_label_1_scores.pt")
        # else:
        label_0_score, label_1_score,  final_motif_0, final_motif_1 = motif_filter(self.whole_dataset, heter_path, self.smiles_set, self.model, 2, self.device)
        # selected_motif = torch.load("checkpoints/motif_selection/"+'BBBP'+"_motif_"+str(self.label)+".pt")

        mask = torch.zeros((1, len(self.smiles_set)), dtype=torch.long).to(self.device)
        
        selected_motif = final_motif_0 if self.label == 0 else final_motif_1

        for i, smiles in enumerate(self.smiles_set):
            if smiles in selected_motif:
                mask[0][i] = 1
        self.motif_mask = mask.bool()
        input("Motif Mask learning done for each class. Showing Class "+str(self.label))
        input(self.motif_mask)
        
        return label_0_score, label_1_score

                
    def load(self, path_dict):
        # Load the model from the path
        self.T_encoder.load_state_dict(torch.load(path_dict['T_encoder']))
        self.pred_node_topo.load_state_dict(torch.load(path_dict['pred_node_topo']))
        self.pred_node_label.load_state_dict(torch.load(path_dict['pred_node_label']))
        self.linear_topo.load_state_dict(torch.load(path_dict['linear_topo']))
        self.linear_label.load_state_dict(torch.load(path_dict['linear_label']))
        self.T_mean.load_state_dict(torch.load(path_dict['T_mean']))
        self.T_var.load_state_dict(torch.load(path_dict['T_var']))

        for param in self.T_encoder.parameters():
            print(param.requires_grad)
        self.T_encoder.eval()
        self.pred_node_topo.eval()
        self.pred_node_label.eval()
        self.linear_topo.eval()
        self.linear_label.eval()
        self.T_mean.eval()
        self.T_var.eval()

    def save(self, path_dict):
        # Save the model to the path
        # torch.save(self.T_encoder.state_dict(), path_dict['T_encoder'])
        torch.save(self.pred_node_topo.state_dict(), path_dict['pred_node_topo'])
        torch.save(self.pred_node_label.state_dict(), path_dict['pred_node_label'])
        torch.save(self.linear_topo.state_dict(), path_dict['linear_topo'])
        torch.save(self.linear_label.state_dict(), path_dict['linear_label'])
        torch.save(self.T_mean.state_dict(), path_dict['T_mean'])
        torch.save(self.T_var.state_dict(), path_dict['T_var'])
        
    def get_motif_score(self):
        
        node_scores = []
        for motif in self.smiles_set:#self.motif_id.keys():
            data = build_graph(motif, None, None)
            data.to(self.device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
            pred, _ = self.model(data.x, data.edge_index, batch)
            # pred = F.softmax(pred, dim=1)
            pred = F.sigmoid(pred)
            node_scores.append(pred[0, 0].item())
        return torch.tensor(node_scores)
    
    def encode_tree(self, tree):
        T_emb = self.T_encoder(self.motif_embedding[tree.x.view(-1)], tree.edge_index, batch=tree.batch, return_embedding=True)
        return T_emb
    
    def encode_graph(self, graph):
        G_emb = self.model(graph.x, graph.edge_index, graph.batch)
        G_emb = self.model.readout_output
        return G_emb


    def train_t_encoder(self, epochs, lr, batch_size, save_path, train_motif_embedding=False):
        """
        Train the teacher encoder (T_encoder) to align tree embeddings with graph embeddings.

        Args:
            epochs (int): Number of epochs to train.
            lr (float): Learning rate for optimizer.
            batch_size (int): Size of mini-batches.
            save_path (str): Path to save the best T_encoder weights.
            train_motif_embedding (bool): Whether to train self.motif_embedding as a parameter.
        """
        # os.makedirs(save_path, exist_ok=True)
        
        # split the dataset into train and test
        train_trees, test_trees = train_test_split(self.trees, test_size=0.2, shuffle=True, random_state=42)
        train_loader = DataLoader(train_trees, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        test_loader = DataLoader(test_trees, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        optimizer = torch.optim.Adam(self.T_encoder.parameters(), lr=lr)
        best_loss = float('inf')
        

        # Show what to optimize
        params = list(self.T_encoder.parameters())
        if train_motif_embedding:
            # Make motif_embedding learnable
            if not isinstance(self.motif_embedding, nn.Parameter):
                self.motif_embedding = nn.Parameter(self.motif_embedding, requires_grad=True)
            params.append(self.motif_embedding)
            print("Training motif_embedding along with T_encoder.")
        else:
            # Ensure motif_embedding is frozen
            self.motif_embedding.requires_grad_(False)
            print("motif_embedding frozen.")

        for epoch in tqdm(range(epochs)):
            total_loss = 0
            # Put T_encoder in train mode
            self.T_encoder.train()
            for t_data, g_data in (train_loader):
                t_data.to(self.device)
                g_data.to(self.device)
                tree_emb = self.encode_tree(t_data)
                graph_emb = self.encode_graph(g_data)
                loss = self.mse_loss(tree_emb, graph_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.T_encoder.eval()
            total_loss = 0
            for t_data, g_data in (test_loader):
                t_data.to(self.device)
                g_data.to(self.device)
                tree_emb = self.encode_tree(t_data)
                graph_emb = self.encode_graph(g_data)
                loss = self.mse_loss(tree_emb, graph_emb)
                total_loss += loss.item()
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.T_encoder.state_dict(), save_path)
                
                
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,  output_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index=None, batch=None, return_embedding=False, classifier=False):
        if classifier:
            return self.lin(x)
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        if batch != None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        if return_embedding:
            return x
        x = self.lin(x)
        
        return x
    
class TGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,  output_channels):
        super(TGCN, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels)
        # self.conv1 = Linear(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, return_embedding=False, classifier=False):
        if classifier:
            return self.lin(x)
        # 1. Obtain node embeddings 
        # print(motif_embedding)
        # x = self.conv1(x)
        if edge_weight != None:
            x = self.conv1(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
        x = x.relu()
        if edge_weight != None:
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv2(x, edge_index)
        x = x.relu()
        if edge_weight != None:
            x = self.conv3(x, edge_index, edge_weight)
        else:
            x = self.conv3(x, edge_index)

        # 2. Readout layer
        if batch != None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        if return_embedding:
            return x
        x = self.lin(x)
        
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_channels, hidden_channels, heads=heads, add_self_loops=False, bias=False)

    def forward(self, x, edge_index, return_weight=False):

        x, edge_tuple = self.conv1(x, edge_index, return_attention_weights=return_weight)

        return x, edge_tuple
    
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
        
        
def get_mol(smiles, addH=False):
    RDLogger.DisableLog('rdApp.*')  
    mol = Chem.MolFromSmiles(smiles)
    if addH == True:
        mol = Chem.AddHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True) # Add clearAromaticFlags to avoid error
    return mol

def get_smiles(mol):
    RDLogger.DisableLog('rdApp.*') 
    smiles = Chem.MolToSmiles(mol)
    return smiles

def sanitize_mol(mol, addH=False):
    try:
        mol = get_mol(get_smiles(mol), addH=addH)
    except:
        return None
    return mol

def sanitize_smiles(smiles, addH=False):
    try:
        mol = get_mol(smiles, addH=addH)
        smiles = get_smiles(mol)

    except:
        return None
    return smiles

# Function to find potential bonding sites
def find_bonding_sites(mol):
    bonding_sites = []
    for atom in mol.GetAtoms():
        # Check if the atom has free valence
        if atom.GetImplicitValence() > 0:
            bonding_sites.append(atom.GetIdx())
    return bonding_sites

def check_bond_feasibility(mol1, mol2, site1, site2):
    # Temporary combining of fragments for checking
    combined_mol = Chem.CombineMols(mol1, mol2)
    editable_mol = Chem.EditableMol(combined_mol)
    
    # Add a bond between chosen sites
    editable_mol.AddBond(site1, mol1.GetNumAtoms() + site2, order=Chem.rdchem.BondType.SINGLE)
    
    # Create the new molecule
    new_mol = editable_mol.GetMol()
    
    # Try sanitizing the molecule; if it fails, the bond is not feasible
    try:
        sanitize_mol(new_mol)
    except:
        return False
    return True

def can_assemble(mol, fragment):
    # Get bonding sites for each fragment
    sites1 = find_bonding_sites(mol)
    sites2 = find_bonding_sites(fragment)
    site_pair = []
    for site1 in sites1:
        for site2 in sites2:
            if check_bond_feasibility(mol, fragment, site1, site2):
                site_pair.append((site1, site2))
    if len(site_pair) > 0:
        return site_pair
    return None