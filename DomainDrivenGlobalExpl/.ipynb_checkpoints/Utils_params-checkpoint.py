import torch
import numpy as np
import sys
from collections import defaultdict
from torch_geometric.loader import DataLoader
from Utils_Train import evaluate_model_prediction, get_masked_graphs_from_list, get_model_prediction
import pdb
import csv

def normalize_ratio(c0, c1, epsilon=1e-8):
    """Calculate normalized ratio difference."""
    r0 = c0 / (c0 + c1 + epsilon)
    r1 = 1 - r0
    return r0, r1

def scale_frequencies(c0, c1, max_c0, max_c1, alpha=0.5):
    """Scale the frequencies within each class."""
    f0_scaled = pow(c0 / max_c0, alpha)
    f1_scaled = pow(c1 / max_c1, alpha)
    return f0_scaled, f1_scaled

def combine_scores(r0, r1, f0_scaled, f1_scaled, beta=1.0, gamma=0.5):
    """Combine ratio difference with scaled frequencies."""
    x0 = beta * r0 + gamma * f0_scaled
    x1 = beta * r1 + gamma * f1_scaled
    return torch.Tensor([x0, x1])

def normalize_motif_score(c0, c1, max_c0, max_c1, alpha=0.5, beta=1.0, gamma=0.5, epsilon=1e-8):
    """Calculate the normalized motif score."""
    r0, r1 = normalize_ratio(c0, c1, epsilon)
    f0_scaled, f1_scaled = scale_frequencies(c0, c1, max_c0, max_c1, alpha)
    return combine_scores(r0, r1, f0_scaled, f1_scaled, beta, gamma)

def calculate_class_totals(df, dataset_name):
    """Calculate the total number of graphs for each class."""
    class_1_graph_total = df[dataset_name].sum()
    class_0_graph_total = len(df) - class_1_graph_total
    return class_0_graph_total, class_1_graph_total

def build_index_class_dict(df, dataset_name, graph_to_motifs, lookup):
    """Build a dictionary with counts of motifs for each class."""
    indx_class_dict = defaultdict(lambda: {0: 1, 1: 1})
    for g, label in zip(df.smiles.tolist(), df[dataset_name].tolist()):
        if g in lookup:
            for indx in graph_to_motifs[g]:
                indx_class_dict[indx][label] += 1
    return indx_class_dict

def calculate_max_counts(indx_class_dict):
    """Calculate maximum counts for normalization."""
    max_c0 = max(indx_class_dict[indx][0] for indx in indx_class_dict.keys())
    max_c1 = max(indx_class_dict[indx][1] for indx in indx_class_dict.keys())
    return max_c0, max_c1

def initialize_parameters(motif_list, indx_class_dict, max_c0, max_c1, alpha=0.5, beta=1.0, gamma=0.5):
    """Initialize the parameters tensor and value counts."""
    parameters = np.zeros((len(motif_list), 2), dtype=np.float32)
    value_counts = np.zeros((len(motif_list), 2), dtype=np.float32)
    
    for i, motif in enumerate(motif_list):
        class_counts = indx_class_dict.get(i, {0: 1, 1: 1})
        c0, c1 = class_counts[0], class_counts[1]

        # Calculate normalized motif score
        normalized_motif_score = normalize_motif_score(c0, c1, max_c0, max_c1, alpha, beta, gamma)
        epsilon = sys.float_info.epsilon
        # normalized_motif_score = torch.clamp((normalized_motif_score-0.5)*1.8, 0.0, 1 - epsilon)
        normalized_motif_score = torch.clamp((normalized_motif_score), 0.0, 1 - epsilon)

        # Calculate log odds and clip values
        parameters[i] = torch.clamp(
            torch.log(normalized_motif_score) - torch.log(1 - normalized_motif_score),
            -3.0, 3.0
        )
        value_counts[i] = np.array([c0, c1], dtype=np.float32)

    parameters_tensor = torch.nn.Parameter(torch.tensor(parameters, dtype=torch.float32), requires_grad=True)
    
    return parameters_tensor, value_counts

def init_parameters(df, dataset_name, motif_list, graph_to_motifs, lookup, alpha=0.5, beta=1.0, gamma=0.5, sigmoid_flag=False, scale = 0.5):
    """Main function to initialize parameters and value counts."""
    class_0_graph_total, class_1_graph_total = calculate_class_totals(df, dataset_name)
    indx_class_dict = build_index_class_dict(df, dataset_name, graph_to_motifs, lookup)
    max_c0, max_c1 = calculate_max_counts(indx_class_dict)
    if sigmoid_flag:
        return initialize_parameters_sigmoid(motif_list, indx_class_dict, max_c0, max_c1, alpha, beta, gamma, scale)
    else:
        return initialize_parameters(motif_list, indx_class_dict, max_c0, max_c1, alpha, beta, gamma)

def initialize_parameters_sigmoid(motif_list, indx_class_dict, max_c0, max_c1, alpha=0.5, beta=1.0, gamma=0.5, scale = 0.5):
    """Initialize the parameters tensor and value counts."""
    parameters = np.zeros((len(motif_list), 2), dtype=np.float32)
    value_counts = np.zeros((len(motif_list), 2), dtype=np.float32)
    
    for i, motif in enumerate(motif_list):
        class_counts = indx_class_dict.get(i, {0: 1, 1: 1})
        c0, c1 = class_counts[0], class_counts[1]

        # Calculate normalized motif score
        normalized_motif_score = normalize_motif_score(c0, c1, max_c0, max_c1, 0.5, 1.0, 0.4)
        epsilon = sys.float_info.epsilon
        normalized_motif_score = torch.clamp((normalized_motif_score-0.5), 0.0, 1 - epsilon)
        # normalized_motif_score = torch.clamp((normalized_motif_score), 0.0, 1 - epsilon)

        # Calculate log odds and clip values
        parameters[i] = torch.clamp(
            torch.log(normalized_motif_score) - torch.log(1 - normalized_motif_score),
            -scale, scale
        )
        value_counts[i] = np.array([c0, c1], dtype=np.float32)

    parameters_tensor = torch.nn.Parameter(torch.tensor(parameters, dtype=torch.float32), requires_grad=True)
    
    return parameters_tensor, value_counts


def get_marginal_importance_of_motifs(loader, lookup_dict, graph_to_motifs, vanilla_model, device):
    motif_weights = vanilla_model.motif_params.detach().cpu()
    result = {}

    for motif_idx, weight in enumerate(motif_weights):
        print(f"{motif_idx} of {motif_weights.shape[0]}")
        
        # Filter graphs in the original loader that contain the motif
        filtered_data = []
        unique_labels = set()
        for batch in loader:
            batch_smiles = batch.smiles  # Assuming `batch` has an attribute `smiles` containing SMILES strings
            for i, smiles in enumerate(batch_smiles):
                if motif_idx in graph_to_motifs[smiles]:
                    filtered_data.append(batch[i])
                    unique_labels.add(batch.y[i].item())
        
        # Create a new DataLoader with the filtered data
        if filtered_data:
            filtered_loader = DataLoader(filtered_data, batch_size=loader.batch_size, shuffle=False)
            
            # Evaluate the model on the original graphs containing the motif
            original_pred,  original_pred_y = evaluate_model_prediction(vanilla_model, filtered_loader, device)
            
            # Apply masking to the graphs containing the motif
            masked_data = get_masked_graphs_from_list(filtered_data, motif_idx, vanilla_model, lookup_dict)
            
            masked_loader = DataLoader(masked_data, batch_size=loader.batch_size, shuffle=False)
            
            # Evaluate the model on the masked graphs
            new_pred, _ = evaluate_model_prediction(vanilla_model, masked_loader, device, original_pred_y)

            original_pred_y = torch.cat(original_pred_y)
            original_pred = torch.stack(original_pred)
            new_pred = torch.stack(new_pred)
            
            # Store the results
            for class_label in [0,1]:
                mask_of_graph_belonging_to_class = (original_pred_y == class_label)
                # input(mask_of_graph_belonging_to_class)
                
                original_pred_of_class = original_pred[mask_of_graph_belonging_to_class]
                # input(original_pred_y)
                new_pred_of_class = new_pred[mask_of_graph_belonging_to_class]
                
                for opred, npred in zip(original_pred_of_class, new_pred_of_class):
                    result[(weight[class_label].item())] = (opred.item(), npred.item(), class_label, motif_idx)
                    # input(result)
        else:
            # If no graphs contain the motif, skip this motif
            print(f"No graphs found containing motif {motif_idx}")
    
    return result




def get_motif_importance_stat(loader, lookup_dict, graph_to_motifs, vanilla_model, device):
    motif_weights = vanilla_model.motif_params.detach().cpu()
    result = defaultdict(list)

    for motif_idx, weight in enumerate(motif_weights):
        print(f"{motif_idx} of {motif_weights.shape[0]}")
        
        # Filter graphs in the original loader that contain the motif
        batch_list = []
        smiles_list= []
        filtered_data = []
        unique_labels = set()
        for batch_id,batch in enumerate(loader):
            batch_smiles = batch.smiles  # Assuming `batch` has an attribute `smiles` containing SMILES strings
            for i, smiles in enumerate(batch_smiles):
                if motif_idx in graph_to_motifs[smiles]:
                    
                    
                    batch_list.append(batch_id)
                    smiles_list.append(i)
                    filtered_data.append(batch[i])
                    unique_labels.add(batch.y[i].item())
                    
        # Create a new DataLoader with the filtered data
        if filtered_data:
            filtered_loader = DataLoader(filtered_data, batch_size=loader.batch_size, shuffle=False)
            
            # Evaluate the model on the original graphs containing the motif
            original_pred, y_label = get_model_prediction(vanilla_model, filtered_loader, device)
            
            # Apply masking to the graphs containing the motif
            masked_data = get_masked_graphs_from_list(filtered_data, motif_idx, vanilla_model, lookup_dict)
            
            masked_loader = DataLoader(masked_data, batch_size=loader.batch_size, shuffle=False)
            
            # Evaluate the model on the masked graphs
            new_pred, _= get_model_prediction(vanilla_model, masked_loader, device)
            
            # input(original_pred)
            # input(new_pred)

            original_pred = torch.stack(original_pred)
            new_pred = torch.stack(new_pred)
            labels = torch.stack(y_label)
            
            for opred, npred, label_y,batch_idx,smile_idx in zip(original_pred, new_pred, labels,batch_list,smiles_list):
            
                result["motif_id"].append(motif_idx)
                result["batch_id"].append(batch_idx)
                result["smile_id"].append(smile_idx)
                for channel_id in range(weight.size(dim=0)):
                    result[f"importance_for_class_{channel_id}"].append(weight[channel_id].item())
                    result[f"sigmoid_importance_for_class_{channel_id}"].append(torch.sigmoid(weight[channel_id]).item())
                for class_id in range(opred.size(dim=0)):
                    result[f"original_logit_class_{class_id}"].append(opred[class_id].item())
                    result[f"new_logit_class_{class_id}"].append(npred[class_id].item())
                result["class_label"].append(label_y.item())
            
                
                
        else:
            # If no graphs contain the motif, skip this motif
            print(f"No graphs found containing motif {motif_idx}")
    
    return result

def save_csv_motif_importance(model, epoch, image_dir, motif_list, image_files, masked_data, csv_file_path):
    # Get the device from the model
    model_device = next(model.parameters()).device
    motif_weights = model.motif_params.detach().cpu()
    csv_data = []  # Collect data for the CSV file
    
    for motif_idx, motif_id in enumerate(motif_list):
        print(f"Processing motif {motif_idx}")
        logit_diff = torch.tensor([[0.0]], device=model_device)
        total_graphs = 0  # To track the total number of graphs across train, val, and test
        
        # Process each dataset: train, val, test
        for dataset_idx, dataset in enumerate(masked_data):
            for graph_idx in dataset[0][motif_idx]:
                total_graphs += 1  # Count graphs
                data = dataset[1][graph_idx].to(model_device)
                
                # Original and perturbed predictions
                original_prediction, _ = model(data.x, data.edge_index, None, data.smiles)
                new_prediction, _ = model(
                    dataset[0][motif_idx][graph_idx].to(model_device), 
                    data.edge_index, 
                    None, 
                    data.smiles
                )
                
                logit_diff += original_prediction - new_prediction
                
                # Collect data for CSV
                
                importance_class_0 = motif_weights[motif_idx].item()
                sigmoid_importance_class_0 = torch.sigmoid(motif_weights[motif_idx]).item()
                csv_data.append([
                    motif_idx,
                    motif_id,
                    graph_idx,
                    importance_class_0,
                    sigmoid_importance_class_0,
                    original_prediction.item(),
                    new_prediction.item(),
                    int(data.y.item())  # Assuming `data.y` contains the class label
                ])
        
    # Write data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "motif_id",
            "motif",
            "graph_id", 
            "importance_for_class_0", 
            "sigmoid_importance_for_class_0", 
            "original_logit_class_0", 
            "new_logit_class_0", 
            "class_label"
        ])
        writer.writerows(csv_data)
