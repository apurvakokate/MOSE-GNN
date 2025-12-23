import CONSTANTS
import pdb
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch.optim import AdamW
import random
import numpy as np
import sys
from collections import defaultdict
import pandas as pd
from Model.Single_channel_gnn import GNNModel
from DataLoader import MolDataset, get_setup_files_with_folds
from Parser import get_parser
import json
import os
import csv
# Training the model and plotting the losses
from Utils.Utils_Train import train_and_evaluate_model, remove_bad_mols, evaluate_model, mae, rmse, compute_pos_weights
from Utils.Utils_plot import plot_losses
from Utils.Utils_params import save_csv_motif_importance, save_csv_motif_importance_optimized
from Utils.Utils_model import compute_deg
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os, wandb
from pathlib import Path

# Choose where you want runs & (optionally) the artifact cache to live
WANDB_DIR = "/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/wandb_runs"

# Make sure they exist and set env vars BEFORE importing/initializing wandb
Path(WANDB_DIR).mkdir(parents=True, exist_ok=True)

os.environ["WANDB_DIR"] = WANDB_DIR

# Optional on clusters without internet:
# os.environ["WANDB_MODE"] = "offline"   # later: `wandb sync ./wandb`
# os.environ["WANDB__SERVICE_WAIT"] = "300"


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
path= args.path

# We dont need these during Vvanilla computation but use it to create dataloader
lookup, motif_list, motif_counts, motif_lengths, motif_class_count, graph_to_motifs, test_data_lookup, test_graph_to_motifs, train_mask_data, val_mask_data, test_mask_data = get_setup_files_with_folds(dataset_name, date_tag, args.fold, args.algorithm, path= path)

if CONSTANTS.DATASET_TYPE[args.dataset_name] == 'BinaryClass':
    num_classes = 2
elif CONSTANTS.DATASET_TYPE[args.dataset_name] == 'Regression':
    num_classes = 1
else:
    raise Exception("Use Muliti Label training code")


if CONSTANTS.DATASET_TYPE[args.dataset_name] == 'Regression':
    # Access training and validation data
    training_data = MolDataset(root=".", split='training',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], normalize = True, mean = None, std = None, lookup = lookup, num_classes = num_classes)
    validation_data = MolDataset(root=".", split='valid',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], normalize = True, mean = training_data.mean, std = training_data.std, lookup = lookup, num_classes = num_classes)
    test_data = MolDataset(root=".", split='test',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], normalize = True, mean = training_data.mean, std = training_data.std, lookup = test_data_lookup, num_classes = num_classes)
    
elif CONSTANTS.DATASET_TYPE[args.dataset_name] == 'BinaryClass':
    training_data = MolDataset(root=".", split='training',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], normalize = False, lookup = lookup, num_classes = num_classes)
    validation_data = MolDataset(root=".", split='valid',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], normalize = False, lookup = lookup, num_classes = num_classes)
    test_data = MolDataset(root=".", split='test',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = CONSTANTS.DATASET_COLUMN[args.dataset_name], normalize = False, lookup = test_data_lookup, num_classes = num_classes)
    
else:
    input("Only Regression and binary classification supported in a single channel")

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
train_loader = DataLoader(training_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True) 
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

print(f"DataLoader ready: {config}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config["model_type"] == "Vanilla":
    model = GNNModel(input_dim = training_data.num_features, 
                      hidden_channels = config["hidden"], 
                      output_dim = 1, 
                      num_layers = config["num_mp_layers"],
                      layer_type = config["layer_type"],
                      use_explainer=False,
                      task_type = args.task_type,
                      )
    
elif config["model_type"] == "SingleChannel":
    params_motif_x_class = torch.full((len(motif_list), 1), args.base_importance).to(device)
    model = GNNModel(input_dim = training_data.num_features, 
                      hidden_channels = config["hidden"], 
                      output_dim = 1,
                      num_layers = config["num_mp_layers"],
                      layer_type = config["layer_type"],
                      use_explainer=True,
                      motif_params = params_motif_x_class,
                      lookup = lookup,
                      task_type = args.task_type,
                      test_lookup = test_data_lookup,
                      unk_importance = args.unk_importance,
                      use_gumbel_softmax = args.use_gumbel_softmax,
                      use_stl = args.use_stl,
                      use_annealing = args.use_annealing, 
                      gumbel_tau = args.gumbel_tau,
                      learn_unknown = args.learn_unknown,
                      use_zero_weight= args.use_zero_weight)
    
    
else:
    '''
    Note: Multi Channel not supported in this case
    '''
    raise Exception("Use Multi Channel file for Muli-Task learning")

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

    
# crit = torch.nn.CrossEntropyLoss()
if args.task_type =='Regression':
    crit = torch.nn.MSELoss()
    class_weights = None
else:
    #Binary Classification
    crit = torch.nn.BCEWithLogitsLoss()
    class_weights = compute_pos_weights(training_data)

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
    
    # (Optional) track gradients/weights every N steps
    # wandb.watch(model, log="all", log_freq=100)

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
                                                                              class_weights = class_weights,
                                                                              patience = args.patience,
                                                                              scheduler = scheduler,
                                                                              clip_grad_norm=True)    
    # image_path = output_dir+f"/explainer/{dataset_name}_losses.png"
    # plot_losses(train_losses, val_losses, dataset_name, image_path)
    # image_path = output_dir+f"/explainer/{dataset_name}_roc-auc.png"
    # plot_losses(train_accs, val_accs, dataset_name, image_path, headers = ["Training Accuracy", "Validation Accuracy"])
    # plot_motif_impact(output_dir+f"/explainer/", f"{dataset_name}_roc-auc.png", dataset_name) 
    
    
model_state = torch.load(output_dir+model_path)
model.load_state_dict(model_state)
if args.task_type == 'Regression':
    EXPERIMENT_RESULTS["Trained_explainations_train_mae"] = mae(model, train_loader, device)
    EXPERIMENT_RESULTS["Trained_explainations_validation_mae"] = mae(model, val_loader, device)
    EXPERIMENT_RESULTS["Trained_explainations_test_mae"] = mae(model, test_loader, device)
    EXPERIMENT_RESULTS["Trained_explainations_train_rmse"] = rmse(model, train_loader, device)
    EXPERIMENT_RESULTS["Trained_explainations_validation_rmse"] = rmse(model, val_loader, device)
    EXPERIMENT_RESULTS["Trained_explainations_test_rmse"] = rmse(model, test_loader, device)
else:
    EXPERIMENT_RESULTS["Trained_explainations_train_rocauc"] = evaluate_model(model, train_loader, device, training_data.num_classes)
    EXPERIMENT_RESULTS["Trained_explainations_validation_rocauc"] = evaluate_model(model, val_loader, device, training_data.num_classes)
    EXPERIMENT_RESULTS["Trained_explainations_test_rocauc"] = evaluate_model(model, test_loader, device, training_data.num_classes)
print("results:",EXPERIMENT_RESULTS)

# Convert dictionary to DataFrame and then export as JSON
pd.DataFrame([EXPERIMENT_RESULTS]).to_json(
    f"{output_dir}/{dataset_name}_classification_result.json", orient='records', lines=True
)

wandb.log(EXPERIMENT_RESULTS)
# Explanation impact visualization
if hasattr(model, 'motif_params'):
    if args.use_zero_weight:
        test_file = f"{output_dir}/{dataset_name}_explanation_result_with_test_zero_weight.csv"
        val_file = f"{output_dir}/{dataset_name}_explanation_result_with_validation_zero_weight.csv"
        train_file = f"{output_dir}/{dataset_name}_explanation_result_with_train_zero_weight.csv"
    else:
        test_file = f"{output_dir}/{dataset_name}_explanation_result_with_test.csv"
        val_file = f"{output_dir}/{dataset_name}_explanation_result_with_validation.csv"
        train_file = f"{output_dir}/{dataset_name}_explanation_result_with_train.csv"

    if not os.path.exists(test_file):
        save_csv_motif_importance_optimized(model, motif_list, [(test_mask_data, test_data)], test_file)
    if not os.path.exists(val_file):
        save_csv_motif_importance_optimized(model, motif_list, [(val_mask_data, validation_data)], val_file)
    if not os.path.exists(train_file):
        save_csv_motif_importance_optimized(model, motif_list, [(train_mask_data, training_data)], train_file)
        
run.finish()