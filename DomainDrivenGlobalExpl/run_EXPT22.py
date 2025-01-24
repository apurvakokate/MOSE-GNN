from tqdm import tqdm
import torch.nn.utils as nn_utils
from IPython.display import display, clear_output
import time

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
from DataLoader import MolDataset, get_setup_files
from Parser import get_parser
import json
import os
import csv
# Training the model and plotting the losses
from Utils_Train import train_and_evaluate_model, remove_bad_mols, evaluate_model, get_masked_graphs_from_list, evaluate_model_prediction
from Utils_plot import plot_losses
from Utils_params import get_marginal_importance_of_motifs, get_motif_importance_stat

from Dual_channel_gin_extend_expl import GNNModel

EXPERIMENT_RESULTS = {}

'''
EXPT 22:
Joint training 
Only 1 single parameter used for both channels. 1-sigmoid(param) is used for channel 0.
'''

args = get_parser()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
date_tag = args.date_tag
dataset_name = args.dataset_name


lookup, motif_list, motif_counts, motif_class_count, graph_to_motifs, test_data_lookup, test_graph_to_motifs = get_setup_files(dataset_name, date_tag)
    


df = pd.read_csv(f"{dataset_name}.csv")

params_motif = torch.full((len(motif_list), 1), args.base_importance)

# Access training and validation data
training_data = MolDataset(root=".", split='training',csv_file=f"{dataset_name}.csv")
validation_data = MolDataset(root=".", split='valid',csv_file=f"{dataset_name}.csv")
test_data = MolDataset(root=".", split='test',csv_file=f"{dataset_name}.csv")

# Removing molecules that cant be parsed by RDkit
training_data = remove_bad_mols(training_data)
validation_data = remove_bad_mols(validation_data)
test_data = remove_bad_mols(test_data)

config = {"num_mp_layers": args.num_mp_layers,
          "hidden":args.hidden,
          "epochs":args.epochs,
          "lr": args.lr,
          "batch_size":args.batch_size,
          "size_reg":args.size_reg,
          "class_reg": args.class_reg,
          "layer_type": args.layer_type,
          "ent_reg":args.ent_reg}


output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(f'{output_dir}/{dataset_name}config.json', 'w') as fp:
    json.dump(config, fp)
    
    
# Create data loaders
batch_size = config["batch_size"]
train_loader = DataLoader(training_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True)
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


vanilla_model = GNNModel(input_dim = training_data.num_features, 
                  hidden_channels = config["hidden"], 
                  output_dim = training_data.num_classes, 
                  num_layers = config["num_mp_layers"],
                  layer_type = config["layer_type"],
                  use_explainer=True,
                  motif_params = params_motif_x_class,
                  lookup = lookup,
                  test_lookup = test_data_lookup)
vanilla_model.to(device)

params_except_w1 = [param for name, param in vanilla_model.named_parameters() if name != 'motif_params']

for param in vanilla_model.parameters():
        param.requires_grad = True

# Now, define the optimizer to only update 'motif_params'
optimizer = AdamW([
    {'params': vanilla_model.motif_params, 'lr': 0.001},  # Only motif_params will be updated
    {'params':params_except_w1}
], config["lr"])

# crit = torch.nn.CrossEntropyLoss()
crit = torch.nn.NLLLoss()

vanilla_model.use_ones = False
model_path = f"/explainer/{dataset_name}_1weighted_best_model.pth"
if os.path.isfile(output_dir+model_path):
    model_state = torch.load(output_dir+model_path)
    vanilla_model.load_state_dict(model_state)
else:
    train_losses, val_losses, train_accs, val_accs = train_and_evaluate_model(vanilla_model, 
                                                                              crit,optimizer,config["epochs"], 
                                                                              train_loader,val_loader, device, config, 
                                                                              output_dir = output_dir+"/explainer/",
                                                                              plot=True,  
                                                                              motif_list=motif_list,
                                                                              ignore_unknowns = args.ignore_unknowns,
                                                                              dataset_name=dataset_name)
    
    image_path = output_dir+f"/explainer/{dataset_name}_losses.png"
    plot_losses(train_losses, val_losses, dataset_name, image_path)
    image_path = output_dir+f"/explainer/{dataset_name}_roc-auc.png"
    plot_losses(train_accs, val_accs, dataset_name, image_path, headers = ["Training Accuracy", "Validation Accuracy"])
    
    
model_state = torch.load(output_dir+model_path)
vanilla_model.load_state_dict(model_state)
EXPERIMENT_RESULTS["Trained_explainations_train_rocauc"] = evaluate_model(vanilla_model, train_loader, device)
EXPERIMENT_RESULTS["Trained_explainations_validation_rocauc"] = evaluate_model(vanilla_model, val_loader, device)
EXPERIMENT_RESULTS["Trained_explainations_test_rocauc"] = evaluate_model(vanilla_model, test_loader, device)

with open(f"{output_dir}/{dataset_name}_classification_result.json", 'w') as fp:
    json.dump(EXPERIMENT_RESULTS, fp)

    
res_test = get_motif_importance_stat(test_loader, vanilla_model.test_lookup, test_graph_to_motifs, vanilla_model, device)


with open(f"{output_dir}/{dataset_name}_explanation_result_with_test.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(res_test.keys())
    writer.writerows(zip(*res_test.values()))
    
res_val = get_motif_importance_stat(val_loader, vanilla_model.lookup, graph_to_motifs, vanilla_model, device)

with open(f"{output_dir}/{dataset_name}_explanation_result_with_validation.csv",mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(res_val.keys())
    writer.writerows(zip(*res_val.values()))
    
res_train = get_motif_importance_stat(train_loader, vanilla_model.lookup, graph_to_motifs, vanilla_model, device)

with open(f"{output_dir}/{dataset_name}_explanation_result_with_train.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(res_train.keys())
    writer.writerows(zip(*res_train.values()))
    

    
    
