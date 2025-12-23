import CONSTANTS
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
from DataLoader import MolDataset, get_setup_files_with_folds
from Parser import get_parser
import json
import os
import csv
from MultiChannel_gnn import GNNModel
# Training the model and plotting the losses
from Utils.Utils_Train import train_and_evaluate_model, remove_bad_mols, evaluate_model, mae, rmse, compute_pos_weights
from Utils.Utils_plot import plot_losses
from Utils.Utils_params import save_csv_motif_importance_optimized # TODO USE save_csv_motif_importance_optimized
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os, wandb
from pathlib import Path

# Choose where you want runs & (optionally) the artifact cache to live
WANDB_DIR = "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/wandb_runs"

# Make sure they exist and set env vars BEFORE importing/initializing wandb
Path(WANDB_DIR).mkdir(parents=True, exist_ok=True)

os.environ["WANDB_DIR"] = WANDB_DIR

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

# We dont need these during Vanilla computation but use it to create dataloader
lookup, motif_list, motif_counts, motif_lengths, motif_class_count, graph_to_motifs, test_data_lookup, test_graph_to_motifs, train_mask_data, val_mask_data, test_mask_data = get_setup_files_with_folds(dataset_name, date_tag, args.fold, args.algorithm)

if CONSTANTS.DATASET_TYPE[args.dataset_name] == 'MultiTask':
    num_classes = len(CONSTANTS.DATASET_COLUMN[args.dataset_name])
else:
    raise Exception("Use Single Channel training code")


training_data = MolDataset(root=".", 
                           split='training',
                           csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", 
                           label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], 
                           normalize = False, 
                           lookup = lookup, 
                           num_classes = num_classes)
validation_data = MolDataset(root=".", 
                             split='valid',\
                             csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", 
                             label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], 
                             normalize = False, 
                             lookup = lookup, 
                             num_classes = num_classes)
test_data = MolDataset(root=".", 
                       split='test',
                       csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", 
                       label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], 
                       normalize = False, 
                       lookup = test_data_lookup, 
                       num_classes = num_classes)

# Removing molecules that cant be parsed by RDkit
training_data = remove_bad_mols(training_data)
validation_data = remove_bad_mols(validation_data)
test_data = remove_bad_mols(test_data)

config = vars(args)


output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(f'{output_dir}/{dataset_name}config.json', 'w') as fp:
    json.dump(config, fp, indent=4)
    
    
# Create data loaders
batch_size = config["batch_size"]
train_loader = DataLoader(training_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True) # TODO check if drop last is required
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

print(f"DataLoader ready: {config}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config["model_type"] == "Vanilla":
    model = GNNModel(input_dim = training_data.num_features, 
                      hidden_channels = config["hidden"], 
                      output_dim = training_data.num_classes, 
                      num_layers = config["num_mp_layers"],
                      layer_type = config["layer_type"],
                      use_explainer=False,
                      task_type = args.task_type)
    
elif config["model_type"] == "MultiChannel":
    params_motif_x_class = torch.full((len(motif_list), training_data.num_classes), args.base_importance).to(device)
    model = GNNModel(input_dim = training_data.num_features, 
                      hidden_channels = config["hidden"], 
                      output_dim = training_data.num_classes, 
                      num_layers = config["num_mp_layers"],
                      layer_type = config["layer_type"],
                      use_explainer=True,
                      motif_params = params_motif_x_class,
                      lookup = lookup,
                      task_type = args.task_type,
                      test_lookup = test_data_lookup,
                      #Additinal arguments for learning unknown motif importance
                      unk_importance = args.unk_importance,

                      learn_unknown = args.learn_unknown)
    
else:
    raise Exception("Model not Supported")

model.to(device)

params_except_w1 = [param for name, param in model.named_parameters() if name != 'motif_params']

for param in model.parameters():
        param.requires_grad = True

# Now, define the optimizer to only update 'motif_params'
if hasattr(model, 'motif_params'):
    optimizer = AdamW([
        {'params': model.motif_params, 'lr': config["expl_lr"]},  # Only motif_params will be updated
        {'params':params_except_w1}
    ], config["lr"])
else:
    optimizer = AdamW([
        {'params':model.parameters()}
    ], config["lr"])

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

if args.task_type =='MultiTask':
    #MultiTask
    crit = torch.nn.BCEWithLogitsLoss(reduction='none') # TODO check reduction
    class_weights_for_positive = compute_pos_weights(training_data)
else:   
    raise Exception("Task not Supported")

# vanilla_model.use_ones = False
# model_path = f"/explainer/{dataset_name}_1weighted_best_model.pth"
model_path = f"/explainer/{dataset_name}_best_model_acc.pth"

final_results_path = f"{output_dir}/{dataset_name}_classification_result.json"

run = wandb.init(
        project="mose-gnn",
        name=f"{dataset_name}-{config['layer_type']}-seed0-fold{args.fold}",
        group=dataset_name,                 # groups runs by dataset
        job_type="train",
        tags=[config["layer_type"], "MOSE", args.task_type],          # whatever helps you filter
        config=config,
    )


if os.path.isfile(final_results_path):
    #Training is complete go to evaluation
    model_state = torch.load(output_dir+model_path)
    model.load_state_dict(model_state)
else:
    os.makedirs(output_dir+"/explainer/", exist_ok=True)
    train_losses, val_losses, train_accs, val_accs = train_and_evaluate_model(model, 
                                                                              crit,optimizer,config["epochs"], 
                                                                              train_loader,val_loader, device, config, 
                                                                              output_dir = output_dir+"/explainer/",
                                                                              plot=False,  
                                                                              motif_list=motif_list,
                                                                              ignore_unknowns = args.ignore_unknowns,
                                                                              dataset_name=dataset_name,
                                                                              train_mask_data = (train_mask_data, training_data),
                                                                              val_mask_data = (val_mask_data, validation_data), 
                                                                              test_mask_data =(test_mask_data, test_data),
                                                                              class_weights = class_weights_for_positive,
                                                                              patience = args.patience,
                                                                              scheduler = scheduler,
                                                                              clip_grad_norm=True)  

    # image_path = output_dir+f"/explainer/{dataset_name}_losses.png"
    # plot_losses(train_losses, val_losses, dataset_name, image_path)
    # image_path = output_dir+f"/explainer/{dataset_name}_roc-auc.png"
    # plot_losses(train_accs, val_accs, dataset_name, image_path, headers = ["Training Accuracy", "Validation Accuracy"])
    
    
model_state = torch.load(output_dir+model_path)
model.load_state_dict(model_state)
EXPERIMENT_RESULTS["Trained_explainations_train_rocauc"] = evaluate_model(model, train_loader, device, training_data.num_classes)
EXPERIMENT_RESULTS["Trained_explainations_validation_rocauc"] = evaluate_model(model, val_loader, device, training_data.num_classes)
EXPERIMENT_RESULTS["Trained_explainations_test_rocauc"] = evaluate_model(model, test_loader, device, training_data.num_classes)
print("results:",EXPERIMENT_RESULTS)
# Convert dictionary to DataFrame and then export as JSON
pd.DataFrame([EXPERIMENT_RESULTS]).to_json(
    f"{output_dir}/{dataset_name}_classification_result.json", orient='records', lines=True
)

# TODO USE save_csv_motif_importance_optimized for Explanation impact visualization
if hasattr(model, 'motif_params'):
    
    test_file = f"{output_dir}/{dataset_name}_explanation_result_with_test_zero_weight.csv"
    val_file = f"{output_dir}/{dataset_name}_explanation_result_with_validation_zero_weight.csv"
    train_file = f"{output_dir}/{dataset_name}_explanation_result_with_train_zero_weight.csv"
    
    if not os.path.exists(test_file):
        save_csv_motif_importance_optimized(model, motif_list, [(test_mask_data, test_data)], test_file, num_classes = training_data.num_classes)
    if not os.path.exists(val_file):
        save_csv_motif_importance_optimized(model, motif_list, [(val_mask_data, validation_data)], val_file, num_classes = training_data.num_classes)
    if not os.path.exists(train_file):
        save_csv_motif_importance_optimized(model, motif_list, [(train_mask_data, training_data)], train_file, num_classes = training_data.num_classes)
        
else:
    test_file = f"{output_dir}/{dataset_name}_{args.algorithm}_test_zero_weight.csv"
    val_file = f"{output_dir}/{dataset_name}_{args.algorithm}_validation_zero_weight.csv"
    train_file = f"{output_dir}/{dataset_name}_{args.algorithm}_train_zero_weight.csv"
    
    if not os.path.exists(test_file):
        save_csv_motif_importance_optimized(model, motif_list, [(test_mask_data, test_data)], test_file, num_classes = training_data.num_classes, vanilla_model= True)
    if not os.path.exists(val_file):
        save_csv_motif_importance_optimized(model, motif_list, [(val_mask_data, validation_data)], val_file, num_classes = training_data.num_classes, vanilla_model= True)
    if not os.path.exists(train_file):
        save_csv_motif_importance_optimized(model, motif_list, [(train_mask_data, training_data)], train_file, num_classes = training_data.num_classes, vanilla_model= True)


    
run.finish()