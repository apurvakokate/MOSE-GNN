from tqdm import tqdm
import torch.nn.utils as nn_utils
import time
import pdb
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch.optim import AdamW
import pickle
import random
import numpy as np
import sys
from collections import defaultdict
from Single_channel_gnn import GNNModel
from DataLoader import MolDataset, get_setup_files_with_folds
from Parser import get_parser
import json
import os
import csv
# Training the model and plotting the losses
from Utils_Train import train_and_evaluate_model, remove_bad_mols, evaluate_model, get_masked_graphs_from_list
from Utils_params import save_csv_motif_importance, save_posthoc_motif_importance
from mage_simplified import MAGE


def save_posthoc_mage_importance(model,mage_model, motif_list, masked_data, csv_file_path="mage/result_global.csv"):
    """
    Evaluate posthoc motif importance using MAGE and save to CSV.
    """
    # Get device
    device = next(model.parameters()).device

    # Get motif mask predictions
    print("Running MAGE get_motif_mask...")
    motif_mask_pred = mage_model.get_motif_score().detach().cpu()
    label_0_score, label_1_score = mage_model.get_motif_mask()
    label_0_score, label_1_score = label_0_score.detach().cpu(), label_1_score.detach().cpu()
    
    csv_data = []

    for motif_idx, motif_id in enumerate(motif_list):
        prob = motif_mask_pred[motif_idx].item()
        label_0_score_motif = label_0_score[motif_idx].item()
        label_1_score_motif = label_1_score[motif_idx].item()
        
        row = [motif_idx, motif_id, prob, label_0_score_motif, label_1_score_motif]
        csv_data.append(row)

    # Unknown motifs (if any)
    row = [-1, "UNK", 0.0, 0, 0, 0]  # Placeholder
    csv_data.append(row)

    headers = ["motif_id", "motif", "mage_prob_importance", "label_0_score", "label_1_score"]

    # Save to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_data)

    print(f"MAGE posthoc evaluation complete. Results saved to {csv_file_path}")



EXPERIMENT_RESULTS = {}

'''
EXPT 12R: 12 Regression
Joint training only
'''

args = get_parser()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
date_tag = args.date_tag
dataset_name = args.dataset_name


lookup, motif_list, motif_counts, motif_lengths, motif_class_count, graph_to_motifs, test_data_lookup, test_graph_to_motifs, train_mask_data, val_mask_data, test_mask_data = get_setup_files_with_folds(dataset_name, date_tag, args.fold, args.algorithm)

dataset_column_dict = {'Mutagenicity':['Mutagenicity'], 
                       'hERG':['hERG'], 
                       'BBBP':['BBBP'],
                       'Lipophilicity':['Lipophilicity'],
                       'esol':['measured log solubility in mols per litre']
                      }


if args.task_type == 'Regression':
    # Access training and validation data
    training_data = MolDataset(root=".", split='training',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[args.dataset_name], normalize = True, mean = None, std = None, lookup = lookup)
    validation_data = MolDataset(root=".", split='valid',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[args.dataset_name], normalize = True, mean = training_data.mean, std = training_data.std, lookup = lookup)
    test_data = MolDataset(root=".", split='test',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[args.dataset_name], normalize = True, mean = training_data.mean, std = training_data.std, lookup = test_data_lookup)
    
elif args.task_type == 'BinaryClass':
    training_data = MolDataset(root=".", split='training',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[args.dataset_name], normalize = False, lookup = lookup)
    validation_data = MolDataset(root=".", split='valid',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[args.dataset_name], normalize = False, lookup = lookup)
    test_data = MolDataset(root=".", split='test',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[args.dataset_name], normalize = False, lookup = test_data_lookup)
    
else:
    input("Only Regression and binary classification supported in a single channel")

# Removing molecules that cant be parsed by RDkit
training_data = remove_bad_mols(training_data)
validation_data = remove_bad_mols(validation_data)
test_data = remove_bad_mols(test_data)

config = vars(args)

# Create data loaders
batch_size = config["batch_size"]
train_loader = DataLoader(training_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True) # TODO check if drop last is required
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

print(f"DataLoader ready: {config}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert config["model_type"] == "SingleChannel"
# model_path = f"/explainer/{dataset_name}_1weighted_best_model.pth"
model_path = f"/explainer/{dataset_name}_best_model_acc.pth"

vanilla_dir = args.vanilla_dir
vanilla_model = GNNModel(input_dim = training_data.num_features, 
                      hidden_channels = config["hidden"], 
                      output_dim = 1, 
                      num_layers = config["num_mp_layers"],
                      layer_type = config["layer_type"],
                      use_explainer=False,
                      task_type = args.task_type)
vanilla_model.to(device)
vanilla_model_state = torch.load(vanilla_dir+model_path)
vanilla_model.load_state_dict(vanilla_model_state)

# Initialize the Mage class
mage = MAGE(gnn=vanilla_model, model=vanilla_model, dataset=training_data+validation_data,
            whole_dataset=training_data+validation_data+test_data, smiles_set=motif_list, 
            hidden_channels=32, output_channels=1, device=device)

# Train a teacher encoder
mage.train_t_encoder(epochs=300, lr=0.0001, batch_size=16, save_path=f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_T_encoder.pth', train_motif_embedding=True)

path_dict = {
    'T_encoder': f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_T_encoder.pth', 
    'pred_node_topo': f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_pred_node_topo.pth', 
    'pred_node_label': f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_pred_node_label.pth', 
    'linear_topo': f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_linear_topo.pth', 
    'linear_label': f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_linear_label.pth', 
    'T_mean': f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_T_mean.pth', 
    'T_var': f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_T_var.pth'}


# mage.train(epochs=100, batch_size=4, lr=0.001, max_iter=5, path_dict=path_dict, t_encoder_path=f'checkpoints/models/{dataset_name}_{args.fold}_{args.layer_type}_T_encoder.pth')
mage.get_motif_mask()
    
# mage.load(path_dict)  # Load pretrained weights

save_posthoc_mage_importance(vanilla_model, mage, motif_list, test_data)

                                    
