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

class MAGE:
    def __init__(self, gnn, model, dataset, smiles_set, hidden_channels, output_channels, device):
        self.gnn = gnn
        self.dataset = dataset
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
        
    def rsample(self, embedding, W_mean, W_var):
        # Compute the mean and log variance
        mu = W_mean(embedding)
        # log_var = W_var(embedding)
        log_var = -torch.abs(W_var(embedding)) # Use the nagetive absolute value of the log variance, with more control
        # Sample from the Gaussian distribution using reparameterization trick
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # `randn_like` generates a tensor with the same size as std, sampled from a standard normal distribution
        z = mu + eps * std  # Reparameterization trick
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence loss
        return z, kl_loss / embedding.size(0)
    
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
                
            set1 = node_motif_map[bond[0]]
            set2 = node_motif_map[bond[1]]

            for node1 in set1:
                for node2 in set2:
                    edge_index.append([node1, node2])
                    edge_index.append([node2, node1])
                
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

    def encode_tree(self, tree):
        self.T_encoder.cpu()
        # print("Start encode the tree!")
        T_emb = self.T_encoder(self.motif_embedding[tree.x.view(-1)], tree.edge_index, batch=tree.batch, return_embedding=True)
        return T_emb
    
    def encode_graph(self, graph):
        G_emb = self.model(graph.x, graph.edge_index, graph.batch)
        G_emb = self.model.readout_output
        return G_emb
    
    def decode_tree(self, trees, graphs, z_tree, max_iter):
        self.T_encoder.cpu()
        tree_list = Batch.to_data_list(trees)
        topo_loss = 0
        label_loss = 0
        emb_loss = 0
        pred_loss = 0
        total_acc_correct = 0
        total_acc_wrong = 0
        total_pred_prob = 0
        count_graph_pred = 0
        count_topo = 0
        count_label = 0
        count_negative = 0
        count_positive = 0
        count_embedding = 0
        
        # For each tree, calculate the loss
        for i, data in enumerate(tree_list):
            self.label = data.data.y.item()
            tree_emb = z_tree[i].unsqueeze(0)
            h_curr = torch.zeros_like(tree_emb)
            if data.x.shape[0] == 1:
                
                topo_pred = self.pred_node_topo(self.linear_topo(tree_emb.add(h_curr.to(self.device))))
                label_pred = self.pred_node_label(self.linear_label(tree_emb.add(h_curr.to(self.device))))

                topo_loss += self.criterion(topo_pred, torch.tensor([1]).to(self.device))
                label_loss += self.criterion(label_pred, data.x[0])
                count_topo += 1
                count_label += 1
                count_positive += 1

                tree_pred_loss, acc_correct, acc_wrong, new_tree, _ = self.sample_tree(tree_emb, max_iter=max_iter)
                pred_loss += tree_pred_loss
                total_acc_correct += acc_correct
                total_acc_wrong += acc_wrong
                continue
            
            # Use BFS to traverse the tree
            queue = deque([(-1, 0)])
            node_neighbors = defaultdict(set)
            for j in range(data.edge_index.shape[1]):
                node_neighbors[data.edge_index[0, j].item()].add(data.edge_index[1, j].item())
                node_neighbors[data.edge_index[1, j].item()].add(data.edge_index[0, j].item())
            visited = set()
            node_order = {}
            curr_x = torch.empty((0, 1), dtype=torch.long).to(self.device)
            curr_edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            while queue:
                # print(f"Queue: {queue}")
                length = len(queue)
                for j in range(length):
                    node = queue.popleft()
                    visited.add(node[1])
                    if node[1] != -2:
                        node_order[node[1]] = len(node_order)
                    if node[0] != -1:
                        # print(curr_x, curr_edge_index)
                        with torch.no_grad():
                            h_curr = self.T_encoder(self.motif_embedding[curr_x.view(-1)], curr_edge_index, return_embedding=True)[node_order[node[0]]]

                    topo_pred = self.pred_node_topo(self.linear_topo(tree_emb.add(h_curr.to(self.device))))
                    # print("topo pred decode_tree node:", node)
                    # input(topo_pred)
                    
                    count_topo += 1
                    
                    # Change label for the last node in the same level
                    if node[1] == -2:
                        topo_loss += self.criterion(topo_pred, torch.tensor([0], device=self.device))
                        count_negative += 1
                    else:
                        topo_loss += self.criterion(topo_pred, torch.tensor([1], device=self.device))
                        count_positive += 1
                        # curr_x = torch.cat((curr_x, data.x[node[1]].unsqueeze(0)), dim=0)
                        curr_x = torch.cat((curr_x, data.x[node[1]].unsqueeze(0).unsqueeze(1)), dim=0)
                        if node[0] != -1:
                            curr_edge_index = torch.cat((curr_edge_index, torch.tensor([[node_order[node[0]], node_order[node[1]]], [node_order[node[1]], node_order[node[0]]]], dtype=torch.long, device=self.device)), dim=1)

                        for neighbor in node_neighbors[node[1]]:
                            if neighbor not in visited:
                                queue.append((node[1], neighbor))
                        queue.append((node[1], -2))
                        
                        label_pred = self.pred_node_label(self.linear_label(tree_emb.add(h_curr.to(self.device))))
                        label_loss += self.criterion(label_pred.cpu(),data.x[node[1]].unsqueeze(0).cpu())
                        count_label += 1
            
            # Freeze teacher forcing
            tree_pred_loss, acc_correct, acc_wrong, new_tree, _ = self.sample_tree(tree_emb, max_iter=max_iter)
            pred_loss += tree_pred_loss
            total_acc_correct += acc_correct
            total_acc_wrong += acc_wrong
            # print(stop)

            smiles, pred_prob, graph_embedding = self.decode_graph(new_tree)
            if smiles:
                batch = torch.zeros(new_tree.x.size(0), dtype=torch.long).to(self.device)
                # with torch.no_grad():
                tree_emb = self.T_encoder(self.motif_embedding[new_tree.x.view(-1)], new_tree.edge_index, batch=batch, return_embedding=True)
                emb_loss += self.mse_loss(tree_emb, graph_embedding)
                count_embedding += 1
        
        return topo_loss / count_topo, label_loss / count_label, pred_loss / len(tree_list), emb_loss/count_embedding, total_acc_correct, total_acc_correct+total_acc_wrong, count_positive, count_negative 
          
    def sample_tree(self, z_tree, max_iter, test=False):
        iter = 0
        queue = deque([(-1, 0)])

        curr_x = torch.empty((0, self.motif_embedding.size(0)), dtype=torch.long).to(self.device)
        curr_edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        curr_edge_weight = torch.empty(0, dtype=torch.float).to(self.device)

        h_curr = torch.zeros_like(z_tree)
        visited = set()
        node_neighbors = defaultdict(list)
        curr_node = -1
        tree_pred_loss = 0
        # print("start")
        while queue and iter < max_iter:
            length = len(queue)
            for j in range(length):
                iter += 1
                node = queue.popleft()
                
                if node[0] != curr_node:
                    print("Huge error!")
                if node[0] != -1:
                    # print("update h_curr")
                    # print(curr_x.size())
                    # print(curr_edge_index)
                    # print(curr_edge_weight)
                    with torch.no_grad():
                        h_curr = self.T_encoder(torch.matmul(curr_x.cpu(), self.motif_embedding.cpu()), curr_edge_index.cpu(), curr_edge_weight.cpu(), return_embedding=True)[node[0]]
                    # batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
                    # h_curr = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, batch=batch, return_embedding=True)
                # with torch.no_grad():
                topo_pred = self.pred_node_topo(self.linear_topo(z_tree.add(h_curr.to(self.device))))
                # print(f"h_curr: {h_curr}")
                if torch.argmax(topo_pred) == 1 or node[0] == -1:                   
                    
                    label_pred = self.pred_node_label(self.linear_label(z_tree.add(h_curr.to(self.device))))
                    if node[0] == -1:
                        softmax_label = self.straight_through_gumbel_softmax(label_pred, temperature=0.1, first_node=True)
                    else:
                        softmax_label = self.straight_through_gumbel_softmax(label_pred, temperature=0.1)
                    # input(softmax_label)
                    
                    if node[0] == -1:
                        curr_x = torch.cat((curr_x, softmax_label), dim=0)
                        motif = self.smiles_set[curr_x[-1].argmax().item()]
                        curr_mol = sanitize_mol(get_mol(motif))
                        batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
                        with torch.no_grad():
                            try:
                                z = torch.matmul(curr_x.cpu(), self.motif_embedding.cpu())
                            except Exception as e:
                                print("❌ matmul error:", e)
                                print("curr_x:", curr_x)
                                pdb.set_trace()

                            tree_emb = self.T_encoder(z, curr_edge_index.cpu(), curr_edge_weight.cpu(), batch=batch.cpu(), return_embedding=True)

                        pred = self.model.classification(tree_emb.to(self.device))
                        tree_pred_loss += self.bce_criterion(pred.squeeze(), torch.tensor(self.label, device=self.device, dtype=torch.float))

                    else:
                        if test:
                            # print("Test!")
                            values, indices = torch.topk(label_pred, 5)
                            # values, indices = torch.topk(softmax_label, 20)
                            # print(values)
                            # print(stop)
                            selected_motif = None
                            for i in range(len(indices[0])):
                                motif = self.smiles_set[indices[0][i].item()]
                                mol = sanitize_mol(get_mol(motif))
                                site_pair = can_assemble(curr_mol, mol)
                                if site_pair:
                                    new_x = torch.nn.functional.one_hot(indices[0][i].view(-1), num_classes=len(self.smiles_set))
                                    curr_x = torch.cat((curr_x, new_x), dim=0)
                                    curr_mol = mol
                                    selected_motif = motif
                                    curr_edge_index = torch.cat((curr_edge_index, torch.tensor([[node[0], node[1]], [node[1], node[0]]], dtype=torch.long, device=self.device)), dim=1)
                                    topo_prob = topo_pred.softmax(1)[0,1]
                                    edge_weight = self.gumbel_softmax_edge_weight(topo_prob)
                                    curr_edge_weight = torch.cat((curr_edge_weight, edge_weight.view(-1)), dim=0)
                                    curr_edge_weight = torch.cat((curr_edge_weight, edge_weight.view(-1)), dim=0)
                                    batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
                                    with torch.no_grad():
                                        tree_emb = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, batch=batch, return_embedding=True)
                                    pred = self.model.classification(tree_emb)
                                    tree_pred_loss += self.bce_criterion(pred.squeeze(), torch.tensor(self.label, device=self.device, dtype=torch.float))
                                    break
                        else:
                            motif = self.smiles_set[softmax_label.argmax().item()]
                            mol = sanitize_mol(get_mol(motif))
                            # site_pair = can_assemble(curr_mol, mol)
                            # if site_pair:
                            curr_x = torch.cat((curr_x, softmax_label), dim=0)
                            curr_mol = mol
                            curr_edge_index = torch.cat((curr_edge_index, torch.tensor([[node[0], node[1]], [node[1], node[0]]], dtype=torch.long, device=self.device)), dim=1)
                            topo_prob = topo_pred.softmax(1)[0,1]

                            edge_weight = self.gumbel_softmax_edge_weight(topo_prob)
                            curr_edge_weight = torch.cat((curr_edge_weight, edge_weight.view(-1)), dim=0)
                            curr_edge_weight = torch.cat((curr_edge_weight, edge_weight.view(-1)), dim=0)
                            batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
                            with torch.no_grad():
                                tree_emb = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, curr_edge_weight, batch=batch, return_embedding=True)
                            pred = self.model.classification(tree_emb)
                            tree_pred_loss += self.bce_criterion(pred.squeeze(), torch.tensor(self.label, device=self.device, dtype=torch.float))
                    
                    if curr_node == -1:
                        queue.append((node[1], curr_x.shape[0]))
                        curr_node += 1
                    else:
                        queue.append((node[0], curr_x.shape[0]))
                    node_neighbors[node[1]].append(curr_x.shape[0] - 1)
                elif torch.argmax(topo_pred) == 0:
                    # if node[0] == -1:
                    curr_node += 1
                    if curr_node < curr_x.shape[0]:
                        visited.add(node[0])
                        queue.append((curr_node, curr_x.shape[0]))
                    else:
                        break
        # if iter >= max_iter:
        #     print("Exceed max_iter!")
        batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
        # print(f"tree_x: {curr_x.size()}")
        with torch.no_grad():
            tree_emb = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, curr_edge_weight, batch=batch, return_embedding=True)
        pred = self.model.classification(tree_emb)
        # print(pred.size())
        tree_pred_loss += self.bce_criterion(pred.squeeze(), torch.tensor(self.label, device=self.device, dtype=torch.float))
        acc_correct, acc_wrong = 0, 0
        if (pred.sigmoid()>0.5) == self.label:
            acc_correct = 1
        else:
            acc_wrong = 1
        new_tree = Data(x=torch.argmax(curr_x, dim=1).view(-1, 1), edge_index=curr_edge_index)
        # print(f"Generated Tree: {new_tree.edge_index}")
        return tree_pred_loss, acc_correct, acc_wrong, new_tree, pred.sigmoid().item()

    def decode_graph(self, tree):
        queue = deque([])
        
        for i in range(0, tree.edge_index.shape[1], 2):
        # for i in range(tree.edge_index.shape[1]-1, -1, -2):
            queue.append((tree.edge_index[0, i].item(), tree.edge_index[1, i].item()))
        curr_mol = None
        visited = set()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
        # bond_types = [Chem.rdchem.BondType.SINGLE]
        # print("Start decoding graph!")
        if not queue:
            motif = self.smiles_set[tree.x[0].item()]
            smiles = sanitize_smiles(motif)
            mol = sanitize_mol(get_mol(smiles))
            data = build_graph(smiles,None,None)
            data.to(self.device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
            pred, _ = self.model(data.x, data.edge_index, batch)
            # pred = F.softmax(pred, dim=1)
            pred = F.sigmoid(pred)
            embedding_output = self.model(data.x, data.edge_index, batch)
            embedding = self.model.readout_output
            return smiles, pred.item(), embedding
            # return smiles, pred[0, self.label].item(), embedding
        while queue:
            node1, node2 = queue.popleft()
            
            motif2 = self.smiles_set[tree.x[node2].item()]
            motif2 = sanitize_mol(get_mol(motif2))
            curr_cand = []
            if curr_mol and not node1 in visited and not node2 in visited:
                print(node1, node2)
                print(visited)
                print(curr_mol)
                print(queue)
                print("Error!")
            elif not curr_mol:
                motif1 = self.smiles_set[tree.x[node1].item()]
                motif1 = sanitize_mol(get_mol(motif1))
                # Create a dictionary store the original motif index of each node in motif1
                atom_motif_id_mapping = {}
                for atom in motif1.GetAtoms():
                    atom_motif_id_mapping[atom.GetIdx()] = node1

                num_atoms = len(atom_motif_id_mapping)
                # Add node in motif2 into atom_motif_id_mapping
                for atom in motif2.GetAtoms():
                    atom_motif_id_mapping[atom.GetIdx()+num_atoms] = node2

                atom_pairs = [(i, j) for i in range(motif1.GetNumAtoms()) for j in range(motif2.GetNumAtoms())]
                for atom1, atom2 in atom_pairs:
                    for bond_type in bond_types:
                        cand = self.combine_motifs(motif1, motif2, atom1, atom2, bond_type)
                        if cand:
                            curr_cand.append(cand)
            else:
                atom_in_motif1 = []
                num_atoms = len(atom_motif_id_mapping)
                # Add node in motif2 into atom_motif_id_mapping
                for atom in motif2.GetAtoms():
                    atom_motif_id_mapping[atom.GetIdx()+num_atoms] = node2
                
                for key, value in atom_motif_id_mapping.items():
                    if value == node1:
                        atom_in_motif1.append(key)

                atom_pairs = [(atom_in_motif1[i], j) for i in range(len(atom_in_motif1)) for j in range(motif2.GetNumAtoms())]
                for atom1, atom2 in atom_pairs:
                    for bond_type in bond_types:
                        cand = self.combine_motifs(curr_mol, motif2, atom1, atom2, bond_type)
                        if cand:
                            curr_cand.append(cand)
            if not curr_cand:
                # print("Error in assemble")
                break
            max_score = 0.0
            # print("hhh")
            for cand in curr_cand:
                data = build_graph(Chem.MolToSmiles(cand), None, None)
                data.to(self.device)
                batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
                pred, _ = self.model(data.x, data.edge_index, batch)
                # sigmoid pred and get the probability of the label
                pred = F.sigmoid(pred)
                # if pred[0, self.label] > max_score:
                #     max_score = pred[0, self.label]
                #     curr_mol = cand
                if pred.item() > max_score:
                    max_score = pred.item()
                    curr_mol = cand
            # print(f"max_score: {max_score}")
            visited.add(node1)
            visited.add(node2)
        try:
            smiles = sanitize_smiles(get_smiles(curr_mol))
        except:
            # print(curr_cand)
            return None, 0, None
        data = build_graph(Chem.MolToSmiles(curr_mol), None, None)
        data.to(self.device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
        pred, _ = self.model(data.x, data.edge_index, batch)
        pred = F.sigmoid(pred)
        embedding_output = self.model(data.x, data.edge_index, batch)
        embedding = self.model.readout_output
        # return smiles, pred[0, self.label].item(), embedding
        return smiles, pred.item(), embedding
    
    def combine_motifs(self, motif1, motif2, atom_idx1, atom_idx2, bond_type):
        combined_mol = Chem.CombineMols(motif1, motif2)
        editable_mol = Chem.EditableMol(combined_mol)
        
        # Add a bond between specified atom indices from each molecule
        num_atoms1 = motif1.GetNumAtoms()
        editable_mol.AddBond(atom_idx1, num_atoms1 + atom_idx2, bond_type)
        
        # Attempt to sanitize the molecule, returns None if unsuccessful
        new_mol = editable_mol.GetMol()

        return sanitize_mol(new_mol)
    
    def straight_through_gumbel_softmax(self, logits, temperature=0.5, first_node=False):
        gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
        # print(self.mask.bool().size())
        if first_node:
            y_soft = torch.softmax(torch.where(self.first_node_mask.bool(), (logits + gumbels) / temperature, torch.tensor(float('-inf'))), dim=-1)
        else:
            y_soft = torch.softmax((logits + gumbels) / temperature, dim=-1)
        y_hard = torch.zeros_like(logits).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)

        # Straight-through estimator trick
        y = y_hard - y_soft.detach() + y_soft
        # y = y_soft
        return y
    
    def gumbel_softmax_edge_weight(self, prob):
        y_soft = prob*2
        y_hard = torch.ones_like(prob)
        y = y_hard - y_soft.detach() + y_soft
        return y
    
    def get_motif_mask(self):
        mask = torch.zeros((1, len(self.smiles_set)), dtype=torch.long).to(self.device)
        for i in range(self.motif_embedding.size(0)):
            emb = self.motif_embedding[i].unsqueeze(0)
            pred = self.model.classification(emb)
            
            # pred = F.softmax(pred, dim=1)
            pred = F.sigmoid(pred)
            
            #if pred[0, self.label] > 0.99:
            if pred[0, 0] > 0.5:
                mask[0][i] = 1
        self.first_node_mask = mask.to(self.device)
        # Not used
#         if os.path.exists("checkpoints/motif_selection/"+self.data_name+"_motif_"+str(self.label)+".pt"):
#             selected_motif = torch.load("checkpoints/motif_selection/"+self.data_name+"_motif_"+str(self.label)+".pt")
#         else:
#             motif_filter(self.whole_dataset, self.data_name, self.smiles_set, self.model, 2, self.device)
#             selected_motif = torch.load("checkpoints/motif_selection/"+self.data_name+"_motif_"+str(self.label)+".pt")

#         mask = torch.zeros((1, len(self.motif_id)), dtype=torch.long).to(self.device)

#         for i, smiles in enumerate(self.motif_id.keys()):
#             if smiles in selected_motif:
#                 mask[0][self.motif_id[smiles]] = 1
#         self.motif_mask = mask.bool()
    
    def train(self, epochs, batch_size, lr, max_iter, path_dict, t_encoder_path):
        # Train the MAGE model
        # For each graph use get_tree to get the tree
        self.get_motif_mask()
        # input(torch.sum(self.mask))
        
        # load the pretrained T_encoder
        self.T_encoder.load_state_dict(torch.load(t_encoder_path))
        # fix parameters of T_encoder
        # for param in self.T_encoder.parameters():
        #     param.requires_grad = False
        
        # check if fix is ok
        for param in self.T_encoder.parameters():
            print(param.requires_grad)
        
        
        # Create the dataloader for the trees
        train_loader = DataLoader(self.trees, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        # Test the dataloader
        optimizer = torch.optim.Adam(list(self.T_encoder.parameters()) + list(self.pred_node_topo.parameters()) + list(self.pred_node_label.parameters()) + list(self.linear_topo.parameters()) + list(self.linear_label.parameters()) + list(self.T_mean.parameters()) + list(self.T_var.parameters()), lr=lr)
        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            total_topo_loss = 0
            total_label_loss = 0
            total_pred_loss = 0
            total_kl_loss = 0
            total_emb_loss = 0
            total_acc = 0
            total_count = 0
            total_graph_pred = 0
            total_graph_sample_count = 0
            for t_data, g_data in tqdm(train_loader):
                t_data.to(self.device)
                g_data.to(self.device)
                
                with torch.no_grad():
                    tree_emb = self.encode_tree(t_data)
                z_tree, kl_tree = self.rsample(tree_emb, self.T_mean, self.T_var)
                
                topo_loss, label_loss, pred_loss, emb_loss, acc, count, count_positive, count_negative = self.decode_tree(t_data, g_data, z_tree, max_iter)

                loss = kl_tree + 10*pred_loss + emb_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_topo_loss += topo_loss.item()
                total_label_loss += label_loss.item()
                total_kl_loss += kl_tree.item()
                if emb_loss != 0:
                    total_emb_loss += emb_loss.item()
                # print(topo_loss, label_loss, pred_loss, emb_loss)
                total_pred_loss += pred_loss.item()
                total_acc += acc
                total_count += count
                
            print(f'Epoch {epoch}, Loss: {total_loss}, Topo Loss: {total_topo_loss}, Label Loss: {total_label_loss}, KL Loss: {total_kl_loss}, Emb Loss: {total_emb_loss}, Pred Loss: {total_pred_loss}, Acc: {total_acc / total_count}, Acc_count: {total_acc}.')
            if total_loss < best_loss:
                best_loss = total_loss
                self.save(path_dict)
                
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
        print("Starting T_encoder training...")
        self.T_encoder.cpu()

        # Put T_encoder in train mode
        self.T_encoder.train()

        # Decide what to optimize
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

        optimizer = torch.optim.Adam(params, lr=lr)

        # Split dataset
        train_trees, test_trees = train_test_split(self.trees, test_size=0.2, shuffle=True, random_state=42)
        train_loader = DataLoader(train_trees, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        test_loader = DataLoader(test_trees, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            total_train_loss = 0.0
            self.T_encoder.train()

            for t_data, g_data in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
                t_data = t_data.to(self.device)
                g_data = g_data.to(self.device)

                tree_emb = self.encode_tree(t_data)  # No torch.no_grad
                graph_emb = self.encode_graph(g_data)

                # Compute loss
                loss = self.mse_loss(tree_emb, graph_emb)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")

            # Evaluate on test set
            total_test_loss = 0.0
            self.T_encoder.eval()
            with torch.no_grad():
                for t_data, g_data in tqdm(test_loader, desc=f"Epoch {epoch} [Test]", leave=False):
                    t_data = t_data.to(self.device)
                    g_data = g_data.to(self.device)

                    tree_emb = self.encode_tree(t_data)
                    graph_emb = self.encode_graph(g_data)

                    test_loss = self.mse_loss(tree_emb, graph_emb)
                    total_test_loss += test_loss.item()

            avg_test_loss = total_test_loss / len(test_loader)
            print(f"Epoch {epoch}: Test Loss = {avg_test_loss:.4f}")

            # Save the best model
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                # Create directory if it does not exist
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.T_encoder.state_dict(), save_path)
                print(f"Best model saved with Test Loss = {best_loss:.4f}")

        print("Training completed.")
                
                
from torch_geometric.nn import GCNConv, GraphConv, GATConv                
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
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Found NaN or Inf in x before conv1!")
        x_cpu = x.cpu()
        edge_index_cpu = edge_index.cpu()
        edge_weight_cpu = edge_weight.cpu() if edge_weight is not None else None
        if edge_weight != None:
            x_cpu = self.conv1(x_cpu, edge_index_cpu, edge_weight_cpu)
        else:
            x_cpu = self.conv1(x_cpu, edge_index_cpu)
        x_cpu = x_cpu.relu()
        if edge_weight != None:
            x_cpu = self.conv2(x_cpu, edge_index_cpu, edge_weight_cpu)
        else:
            x_cpu = self.conv2(x_cpu, edge_index_cpu)
        x_cpu = x_cpu.relu()
        if edge_weight != None:
            x_cpu = self.conv3(x_cpu, edge_index_cpu, edge_weight_cpu)
        else:
            x_cpu = self.conv3(x_cpu, edge_index_cpu)

        # 2. Readout layer
        if batch != None:
            x_cpu = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        if return_embedding:
            return x_cpu
        x_cpu = self.lin(x_cpu)
        
        return x_cpu
    
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