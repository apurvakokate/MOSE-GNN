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
from DataLoader import MolDataset, get_setup_files, get_setup_files_with_folds
from Parser import get_parser
import json
import os
import csv
# Training the model and plotting the losses
from Utils_Train import train_and_evaluate_model, remove_bad_mols, evaluate_model, get_masked_graphs_from_list, evaluate_model_prediction, mae, rmse, plot_and_save_distribution
from Utils_plot import plot_losses
from Utils_params import get_marginal_importance_of_motifs, get_motif_importance_stat, save_csv_motif_importance



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
                       'esol':['measured log solubility in mols per litre']}

if args.task_type == 'Regression':
    # Access training and validation data
    training_data = MolDataset(root=".", split='training',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[dataset_name], task_type = args.task_type, normalize = True, mean = None, std = None)
    validation_data = MolDataset(root=".", split='valid',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[dataset_name], task_type = args.task_type, normalize = True, mean = training_data.mean, std = training_data.std)
    test_data = MolDataset(root=".", split='test',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[dataset_name], task_type = args.task_type, normalize = True, mean = training_data.mean, std = training_data.std)
    
else:
    training_data = MolDataset(root=".", split='training',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[dataset_name], task_type = args.task_type, normalize = False)
    validation_data = MolDataset(root=".", split='valid',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[dataset_name], task_type = args.task_type, normalize = False)
    test_data = MolDataset(root=".", split='test',csv_file=f"datasets/FOLDS/{dataset_name}_{args.fold}.csv", label_col = dataset_column_dict[dataset_name], task_type = args.task_type, normalize = False)

# Removing molecules that cant be parsed by RDkit
training_data = remove_bad_mols(training_data)
validation_data = remove_bad_mols(validation_data)
test_data = remove_bad_mols(test_data)

config = {"model_type": args.model_type,
          "num_mp_layers": args.num_mp_layers,
          "hidden":args.hidden,
          "epochs":args.epochs,
          "lr": args.lr,
          "dataset":dataset_name,
          "algorithm": args.algorithm,
          "fold": args.fold,
          "batch_size":args.batch_size,
          "size_reg":args.size_reg,
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
from Single_channel_gin import GNNModel
if config["model_type"] == "Vanilla":
    # model = GNNModel(input_dim = training_data.num_features, 
    #                   hidden_channels = config["hidden"], 
    #                   output_dim = training_data.num_classes, 
    #                   num_layers = config["num_mp_layers"],
    #                   layer_type = config["layer_type"],
    #                   task_type = args.task_type)
    model = GNNModel(input_dim = training_data.num_features, 
                      hidden_channels = config["hidden"], 
                      output_dim = 1, #training_data.num_classes, 
                      num_layers = config["num_mp_layers"],
                      layer_type = config["layer_type"],
                      use_explainer=False,
                      task_type = args.task_type)
    
elif config["model_type"] == "SingleChannel":
    # from Single_channel_gin import GNNModel
    params_motif_x_class = torch.full((len(motif_list), 1), args.base_importance)
    model = GNNModel(input_dim = training_data.num_features, 
                      hidden_channels = config["hidden"], 
                      output_dim = 1, #training_data.num_classes, 
                      num_layers = config["num_mp_layers"],
                      layer_type = config["layer_type"],
                      use_explainer=True,
                      motif_params = params_motif_x_class,
                      lookup = lookup,
                      task_type = args.task_type,
                      test_lookup = test_data_lookup)
    
else:
    '''
    Note: Multi Channel not supported in this case
    '''
    raise Exception("Multi Channel requires single channel")

model.to(device)

params_except_w1 = [param for name, param in model.named_parameters() if name != 'motif_params']

for param in model.parameters():
        param.requires_grad = True

# Now, define the optimizer to only update 'motif_params'
if hasattr(model, 'motif_params'):
    optimizer = AdamW([
        {'params': model.motif_params, 'lr': 0.001},  # Only motif_params will be updated
        {'params':params_except_w1}
    ], config["lr"])
else:
    optimizer = AdamW([
        {'params':model.parameters()}
    ], config["lr"])

# crit = torch.nn.CrossEntropyLoss()
if args.task_type =='Regression':
    crit = torch.nn.MSELoss()
else:
    #Binary Classification
    crit = torch.nn.BCEWithLogitsLoss()

# vanilla_model.use_ones = False
model_path = f"/explainer/{dataset_name}_1weighted_best_model.pth"
if os.path.isfile(output_dir+model_path):
    model_state = torch.load(output_dir+model_path)
    model.load_state_dict(model_state)
else:
    train_losses, val_losses, train_accs, val_accs = train_and_evaluate_model(model, 
                                                                              crit,optimizer,config["epochs"], 
                                                                              train_loader,val_loader, device, config, 
                                                                              output_dir = output_dir+"/explainer/",
                                                                              plot=True,  
                                                                              motif_list=motif_list,
                                                                              ignore_unknowns = args.ignore_unknowns,
                                                                              dataset_name=dataset_name,
                                                                              train_mask_data = (train_mask_data, training_data),
                                                                              val_mask_data = (val_mask_data, validation_data), 
                                                                              test_mask_data =(test_mask_data, test_data))    
    image_path = output_dir+f"/explainer/{dataset_name}_losses.png"
    plot_losses(train_losses, val_losses, dataset_name, image_path)
    image_path = output_dir+f"/explainer/{dataset_name}_roc-auc.png"
    plot_losses(train_accs, val_accs, dataset_name, image_path, headers = ["Training Accuracy", "Validation Accuracy"])
    
    
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

# Explanation impact visualization
if hasattr(model, 'motif_params'):   
    save_csv_motif_importance(model, 0, "", motif_list, [], [(test_mask_data, test_data)], f"{output_dir}/{dataset_name}_explanation_result_with_test.csv")
    save_csv_motif_importance(model, 0, "", motif_list, [], [(val_mask_data, validation_data)], f"{output_dir}/{dataset_name}_explanation_result_with_validation.csv")
    save_csv_motif_importance(model, 0, "", motif_list, [], [(train_mask_data, training_data)], f"{output_dir}/{dataset_name}_explanation_result_with_train.csv")
#     res_test = plot_and_save_distribution(model, 0, "", motif_list, [], [(test_mask_data, test_data)])
# #     res_test = get_motif_importance_stat(test_loader, vanilla_model.test_lookup, test_graph_to_motifs, vanilla_model, device)
    
# #     compute_motif_impact(mask_data, model, model_device)

#     with open(f"{output_dir}/{dataset_name}_explanation_result_with_test.csv", mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(res_test.keys())
#         writer.writerows(zip(*res_test.values()))

# #     res_val = get_motif_importance_stat(val_loader, vanilla_model.lookup, graph_to_motifs, vanilla_model, device)
#     res_test = plot_and_save_distribution(model, 0, "", motif_list, [], [(val_mask_data, validation_data)])

#     with open(f"{output_dir}/{dataset_name}_explanation_result_with_validation.csv",mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(res_val.keys())
#         writer.writerows(zip(*res_val.values()))
    
# #     res_train = get_motif_importance_stat(train_loader, vanilla_model.lookup, graph_to_motifs, vanilla_model, device)
#     res_train = plot_and_save_distribution(model, 0, "", motif_list, [], [(train_mask_data, training_data)])
#     with open(f"{output_dir}/{dataset_name}_explanation_result_with_train.csv", mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(res_train.keys())
#         writer.writerows(zip(*res_train.values()))
    

    
    
