from tqdm import tqdm
import torch.nn.utils as nn_utils
from IPython.display import display, clear_output
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
import pandas as pd
from Single_channel_gnn import GNNModel
from DataLoader import MolDataset, get_setup_files, get_setup_files_with_folds
from Parser import get_parser
import json
import os
import csv
# Training the model and plotting the losses
from Utils_Train import train_and_evaluate_model, remove_bad_mols, evaluate_model, get_masked_graphs_from_list
from Utils_params import save_posthoc_motif_importance



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

config = {"model_type": args.model_type,
          "num_mp_layers": args.num_mp_layers,
          "hidden":args.hidden,
          "epochs":args.epochs,
          "expl_lr": args.expl_lr,
          "lr": args.lr,
          "dataset":dataset_name,
          "algorithm": args.algorithm,
          "fold": args.fold,
          "batch_size":args.batch_size,
          "size_reg":args.size_reg,
          "date_tag": date_tag,
          "class_reg": args.class_reg,
          "layer_type": args.layer_type,
          "ent_reg":args.ent_reg,
          "task": args.task_type}


output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(f'{output_dir}/{dataset_name}config.json', 'w') as fp:
    json.dump(config, fp)
    
# Create data loaders
batch_size = config["batch_size"]
train_loader = DataLoader(training_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True) # TODO check if drop last is required
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

print(f"DataLoader ready: {config}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert config["model_type"] == "SingleChannel"

params_motif_x_class = torch.full((len(motif_list), 1), args.base_importance)
model = GNNModel(input_dim = training_data.num_features, 
                  hidden_channels = config["hidden"], 
                  output_dim = 1,
                  num_layers = config["num_mp_layers"],
                  layer_type = config["layer_type"],
                  use_explainer=True,
                  motif_params = params_motif_x_class,
                  lookup = lookup,
                  task_type = args.task_type,
                  test_lookup = test_data_lookup)
    
model_path = f"/explainer/{dataset_name}_1weighted_best_model.pth"

model.to(device)
model_state = torch.load(output_dir+model_path)
model.load_state_dict(model_state)

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

# Explanation impact visualization
if hasattr(model, 'motif_params'):  
    # Define file paths
    test_file = f"{output_dir}/{dataset_name}_vanilla_impact_withposthoclr05_g(ent80)_p(sizebydatamodel2)_train_val_test.csv"

    # Save only if the file does not exist
    if not os.path.exists(test_file):
        save_posthoc_motif_importance(model, motif_list, [(train_mask_data, training_data, train_loader),(val_mask_data, validation_data, val_loader),(test_mask_data, test_data, test_loader)], test_file, vanilla_model= vanilla_model, lookup = lookup, test_lookup = test_data_lookup, task_type = args.task_type, dataset_name = dataset_name, model_type = config["layer_type"])
