import torch
import numpy as np
import sys
from collections import defaultdict
from torch_geometric.loader import DataLoader
from Utils_Train import get_masked_graphs_from_list
import pdb
import csv
from Explainer import Explainer
from GNNExplainer import GNNExplainer
from PGExplainer import PGExplainer
from tqdm import tqdm
import torch.nn.functional as F

# def normalize_ratio(c0, c1, epsilon=1e-8):
#     """Calculate normalized ratio difference."""
#     r0 = c0 / (c0 + c1 + epsilon)
#     r1 = 1 - r0
#     return r0, r1

# def scale_frequencies(c0, c1, max_c0, max_c1, alpha=0.5):
#     """Scale the frequencies within each class."""
#     f0_scaled = pow(c0 / max_c0, alpha)
#     f1_scaled = pow(c1 / max_c1, alpha)
#     return f0_scaled, f1_scaled

# def combine_scores(r0, r1, f0_scaled, f1_scaled, beta=1.0, gamma=0.5):
#     """Combine ratio difference with scaled frequencies."""
#     x0 = beta * r0 + gamma * f0_scaled
#     x1 = beta * r1 + gamma * f1_scaled
#     return torch.Tensor([x0, x1])

# def normalize_motif_score(c0, c1, max_c0, max_c1, alpha=0.5, beta=1.0, gamma=0.5, epsilon=1e-8):
#     """Calculate the normalized motif score."""
#     r0, r1 = normalize_ratio(c0, c1, epsilon)
#     f0_scaled, f1_scaled = scale_frequencies(c0, c1, max_c0, max_c1, alpha)
#     return combine_scores(r0, r1, f0_scaled, f1_scaled, beta, gamma)

# def calculate_class_totals(df, dataset_name):
#     """Calculate the total number of graphs for each class."""
#     class_1_graph_total = df[dataset_name].sum()
#     class_0_graph_total = len(df) - class_1_graph_total
#     return class_0_graph_total, class_1_graph_total

# def build_index_class_dict(df, dataset_name, graph_to_motifs, lookup):
#     """Build a dictionary with counts of motifs for each class."""
#     indx_class_dict = defaultdict(lambda: {0: 1, 1: 1})
#     for g, label in zip(df.smiles.tolist(), df[dataset_name].tolist()):
#         if g in lookup:
#             for indx in graph_to_motifs[g]:
#                 indx_class_dict[indx][label] += 1
#     return indx_class_dict

# def calculate_max_counts(indx_class_dict):
#     """Calculate maximum counts for normalization."""
#     max_c0 = max(indx_class_dict[indx][0] for indx in indx_class_dict.keys())
#     max_c1 = max(indx_class_dict[indx][1] for indx in indx_class_dict.keys())
#     return max_c0, max_c1

# def initialize_parameters(motif_list, indx_class_dict, max_c0, max_c1, alpha=0.5, beta=1.0, gamma=0.5):
#     """Initialize the parameters tensor and value counts."""
#     parameters = np.zeros((len(motif_list), 2), dtype=np.float32)
#     value_counts = np.zeros((len(motif_list), 2), dtype=np.float32)
    
#     for i, motif in enumerate(motif_list):
#         class_counts = indx_class_dict.get(i, {0: 1, 1: 1})
#         c0, c1 = class_counts[0], class_counts[1]

#         # Calculate normalized motif score
#         normalized_motif_score = normalize_motif_score(c0, c1, max_c0, max_c1, alpha, beta, gamma)
#         epsilon = sys.float_info.epsilon
#         # normalized_motif_score = torch.clamp((normalized_motif_score-0.5)*1.8, 0.0, 1 - epsilon)
#         normalized_motif_score = torch.clamp((normalized_motif_score), 0.0, 1 - epsilon)

#         # Calculate log odds and clip values
#         parameters[i] = torch.clamp(
#             torch.log(normalized_motif_score) - torch.log(1 - normalized_motif_score),
#             -3.0, 3.0
#         )
#         value_counts[i] = np.array([c0, c1], dtype=np.float32)

#     parameters_tensor = torch.nn.Parameter(torch.tensor(parameters, dtype=torch.float32), requires_grad=True)
    
#     return parameters_tensor, value_counts

# def init_parameters(df, dataset_name, motif_list, graph_to_motifs, lookup, alpha=0.5, beta=1.0, gamma=0.5, sigmoid_flag=False, scale = 0.5):
#     """Main function to initialize parameters and value counts."""
#     class_0_graph_total, class_1_graph_total = calculate_class_totals(df, dataset_name)
#     indx_class_dict = build_index_class_dict(df, dataset_name, graph_to_motifs, lookup)
#     max_c0, max_c1 = calculate_max_counts(indx_class_dict)
#     if sigmoid_flag:
#         return initialize_parameters_sigmoid(motif_list, indx_class_dict, max_c0, max_c1, alpha, beta, gamma, scale)
#     else:
#         return initialize_parameters(motif_list, indx_class_dict, max_c0, max_c1, alpha, beta, gamma)

# def initialize_parameters_sigmoid(motif_list, indx_class_dict, max_c0, max_c1, alpha=0.5, beta=1.0, gamma=0.5, scale = 0.5):
#     """Initialize the parameters tensor and value counts."""
#     parameters = np.zeros((len(motif_list), 2), dtype=np.float32)
#     value_counts = np.zeros((len(motif_list), 2), dtype=np.float32)
    
#     for i, motif in enumerate(motif_list):
#         class_counts = indx_class_dict.get(i, {0: 1, 1: 1})
#         c0, c1 = class_counts[0], class_counts[1]

#         # Calculate normalized motif score
#         normalized_motif_score = normalize_motif_score(c0, c1, max_c0, max_c1, 0.5, 1.0, 0.4)
#         epsilon = sys.float_info.epsilon
#         normalized_motif_score = torch.clamp((normalized_motif_score-0.5), 0.0, 1 - epsilon)
#         # normalized_motif_score = torch.clamp((normalized_motif_score), 0.0, 1 - epsilon)

#         # Calculate log odds and clip values
#         parameters[i] = torch.clamp(
#             torch.log(normalized_motif_score) - torch.log(1 - normalized_motif_score),
#             -scale, scale
#         )
#         value_counts[i] = np.array([c0, c1], dtype=np.float32)

#     parameters_tensor = torch.nn.Parameter(torch.tensor(parameters, dtype=torch.float32), requires_grad=True)
    
#     return parameters_tensor, value_counts


# def get_marginal_importance_of_motifs(loader, lookup_dict, graph_to_motifs, vanilla_model, device):
#     motif_weights = vanilla_model.motif_params.detach().cpu()
#     result = {}

#     for motif_idx, weight in enumerate(motif_weights):
#         print(f"{motif_idx} of {motif_weights.shape[0]}")
        
#         # Filter graphs in the original loader that contain the motif
#         filtered_data = []
#         unique_labels = set()
#         for batch in loader:
#             batch_smiles = batch.smiles  # Assuming `batch` has an attribute `smiles` containing SMILES strings
#             for i, smiles in enumerate(batch_smiles):
#                 if motif_idx in graph_to_motifs[smiles]:
#                     filtered_data.append(batch[i])
#                     unique_labels.add(batch.y[i].item())
        
#         # Create a new DataLoader with the filtered data
#         if filtered_data:
#             filtered_loader = DataLoader(filtered_data, batch_size=loader.batch_size, shuffle=False)
            
#             # Evaluate the model on the original graphs containing the motif
#             original_pred,  original_pred_y = evaluate_model_prediction(vanilla_model, filtered_loader, device)
            
#             # Apply masking to the graphs containing the motif
#             masked_data = get_masked_graphs_from_list(filtered_data, motif_idx, vanilla_model, lookup_dict)
            
#             masked_loader = DataLoader(masked_data, batch_size=loader.batch_size, shuffle=False)
            
#             # Evaluate the model on the masked graphs
#             new_pred, _ = evaluate_model_prediction(vanilla_model, masked_loader, device, original_pred_y)

#             original_pred_y = torch.cat(original_pred_y)
#             original_pred = torch.stack(original_pred)
#             new_pred = torch.stack(new_pred)
            
#             # Store the results
#             for class_label in [0,1]:
#                 mask_of_graph_belonging_to_class = (original_pred_y == class_label)
#                 # input(mask_of_graph_belonging_to_class)
                
#                 original_pred_of_class = original_pred[mask_of_graph_belonging_to_class]
#                 # input(original_pred_y)
#                 new_pred_of_class = new_pred[mask_of_graph_belonging_to_class]
                
#                 for opred, npred in zip(original_pred_of_class, new_pred_of_class):
#                     result[(weight[class_label].item())] = (opred.item(), npred.item(), class_label, motif_idx)
#                     # input(result)
#         else:
#             # If no graphs contain the motif, skip this motif
#             print(f"No graphs found containing motif {motif_idx}")
    
#     return result




# def get_motif_importance_stat(loader, lookup_dict, graph_to_motifs, vanilla_model, device):
#     motif_weights = vanilla_model.motif_params.detach().cpu()
#     result = defaultdict(list)

#     for motif_idx, weight in enumerate(motif_weights):
#         print(f"{motif_idx} of {motif_weights.shape[0]}")
        
#         # Filter graphs in the original loader that contain the motif
#         batch_list = []
#         smiles_list= []
#         filtered_data = []
#         unique_labels = set()
#         for batch_id,batch in enumerate(loader):
#             batch_smiles = batch.smiles  # Assuming `batch` has an attribute `smiles` containing SMILES strings
#             for i, smiles in enumerate(batch_smiles):
#                 if motif_idx in graph_to_motifs[smiles]:
                    
                    
#                     batch_list.append(batch_id)
#                     smiles_list.append(i)
#                     filtered_data.append(batch[i])
#                     unique_labels.add(batch.y[i].item())
                    
#         # Create a new DataLoader with the filtered data
#         if filtered_data:
#             filtered_loader = DataLoader(filtered_data, batch_size=loader.batch_size, shuffle=False)
            
#             # Evaluate the model on the original graphs containing the motif
#             original_pred, y_label = get_model_prediction(vanilla_model, filtered_loader, device)
            
#             # Apply masking to the graphs containing the motif
#             masked_data = get_masked_graphs_from_list(filtered_data, motif_idx, vanilla_model, lookup_dict)
            
#             masked_loader = DataLoader(masked_data, batch_size=loader.batch_size, shuffle=False)
            
#             # Evaluate the model on the masked graphs
#             new_pred, _= get_model_prediction(vanilla_model, masked_loader, device)
            
#             # input(original_pred)
#             # input(new_pred)

#             original_pred = torch.stack(original_pred)
#             new_pred = torch.stack(new_pred)
#             labels = torch.stack(y_label)
            
#             for opred, npred, label_y,batch_idx,smile_idx in zip(original_pred, new_pred, labels,batch_list,smiles_list):
            
#                 result["motif_id"].append(motif_idx)
#                 result["batch_id"].append(batch_idx)
#                 result["smile_id"].append(smile_idx)
#                 for channel_id in range(weight.size(dim=0)):
#                     result[f"importance_for_class_{channel_id}"].append(weight[channel_id].item())
#                     result[f"sigmoid_importance_for_class_{channel_id}"].append(torch.sigmoid(weight[channel_id]).item())
#                 for class_id in range(opred.size(dim=0)):
#                     result[f"original_logit_class_{class_id}"].append(opred[class_id].item())
#                     result[f"new_logit_class_{class_id}"].append(npred[class_id].item())
#                 result["class_label"].append(label_y.item())
            
                
                
#         else:
#             # If no graphs contain the motif, skip this motif
#             print(f"No graphs found containing motif {motif_idx}")
    
#     return result

def save_posthoc_motif_importance(model, motif_list, masked_data, csv_file_path, lookup=None, test_lookup=None, task_type="BinaryClass", dataset_name = None, model_type = None):
    '''
    Example usage save_csv_motif_importance(model, motif_list, [(train_mask_data, training_data)], train_file, vanilla_model= vanilla_model, lookup = test_data_lookup)
    '''
    
    # Get the device from the model
    model_device = next(model.parameters()).device
    csv_data = []  # Collect data for the CSV file
    if task_type == 'BinaryClass':
        expl_mode = 'binary_classification' 
    elif task_type == 'Regression':
        expl_mode = 'regression'
    else:
        raise Exception('PostHoc not implemented for this setting')
    
    if dataset_name == "Lipophilicity" and model_type == "GATConv":
        pgex_edge_size = 0.0001
    elif dataset_name == "hERG" and model_type == "GCNConv":
        pgex_edge_size = 0.0001
    else:
        pgex_edge_size = 0.0003
        
    # Initialize GNNExplainer
    gnn_explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(lr=0.05, epochs=150, edge_ent=8.0),
        explanation_type='phenomenon',
        node_mask_type='object',
        edge_mask_type=None,
        model_config=dict(
            mode=expl_mode,
            task_level='graph',
            return_type='raw',
        ),
    )
    pgex_epoch = 50
    pg_explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=pgex_epoch, lr = 0.05, edge_size=pgex_edge_size), # edge_size=0.0001,edge_ent=0.2
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=dict(
                mode=expl_mode,
                task_level='graph',
                return_type='raw',
            ),
        )
    # Train against a variety of node-level or graph-level predictions:
    for epoch in tqdm(range(pgex_epoch)):
        for data in masked_data[0][2]: #train_loader
            data = data.to(model_device)
            if task_type == 'BinaryClass':
                loss = pg_explainer.algorithm.train(epoch, model, data.x, data.edge_index,
                                                 target=data.y, batch=data.batch)
            elif task_type == 'Regression':
                loss = pg_explainer.algorithm.train(epoch, model, data.x, data.edge_index,
                                             target=data.y.float(), batch=data.batch)

    # Initialize post-hoc weights (same size as the number of motifs)
    gnnex_weights = torch.full((len(motif_list)+1, 1), 0.0)
    pgex_weights = torch.full((len(motif_list)+1, 1), 0.0)

    # Track counts for averaging
    motif_counts = torch.full((len(motif_list)+1, 1), 0)

    for dataset_idx, dataset in enumerate(masked_data): # train val and test samples
        for data in dataset[2]: #loader
            data = data.to(model_device) #original data list

            if task_type == 'BinaryClass':
                target = data.y
            elif task_type == 'Regression':
                target = data.y.float()

            # Run explainer on the graph
            gnnex_explanation = gnn_explainer(data.x, data.edge_index, target = target, batch=data.batch)
            pgex_explanation = pg_explainer(data.x, data.edge_index, target = target, batch=data.batch)


            gnnex_node_importance = gnnex_explanation.node_mask.squeeze().detach().cpu()

            edge_importance = pgex_explanation.edge_mask.detach().cpu()
            edge_index = data.edge_index.cpu()
            pgex_node_importance = torch.zeros(data.num_nodes)

            # Sum edge importance for connected edges per node
            # for i, (src, dst) in enumerate(edge_index.t()):
            #     importance = edge_importance[i].item()
            #     node_importance[src] += importance
            #     node_importance[dst] += importance  # Consider both directions
            src, dst = edge_index[0], edge_index[1]
            pgex_node_importance.scatter_add_(0, src, edge_importance)
            pgex_node_importance.scatter_add_(0, dst, edge_importance)
            import torch_scatter
            degree = torch_scatter.scatter_add(torch.ones_like(edge_index[0]), edge_index[0])
            pgex_node_importance = pgex_node_importance / (2 * degree.clamp(min=1))


            # print(gnnex_node_importance)
            # input(pgex_node_importance)


            # Accumulate importance scores per motif
            batch = data.batch.cpu().numpy()
            num_graphs = int(data.batch.max()) + 1

            for graph_idx in range(num_graphs):
                # Get the SMILES for this subgraph
                graph_smiles = data.smiles[graph_idx]

                # Get node indices belonging to this subgraph
                subgraph_nodes = np.where(batch == graph_idx)[0]

                # Get motif lookup for this specific SMILES
                motif_lookup = lookup[graph_smiles] if dataset_idx != 2 else test_lookup[graph_smiles]

                # Process nodes in this subgraph
                for global_node in subgraph_nodes:
                    # Convert to local node index (original graph's node numbering)
                    local_node = int(global_node - subgraph_nodes[0])

                    if local_node in motif_lookup:
                        motif, motif_idx = motif_lookup[local_node]
                        if motif_idx is not None:
                            gnnex_weights[motif_idx] += gnnex_node_importance[global_node].item()
                            pgex_weights[motif_idx] += pgex_node_importance[global_node].item()
                            motif_counts[motif_idx] += 1
                        else:
                            gnnex_weights[len(motif_list)] += gnnex_node_importance[global_node].item()
                            pgex_weights[len(motif_list)] += pgex_node_importance[global_node].item()
                            motif_counts[len(motif_list)] += 1



    # Compute the average node importance per motif
    gnnex_weights /= motif_counts.clamp(min=1)  # Avoid division by zero
    pgex_weights /= motif_counts.clamp(min=1)  # Avoid division by zero
            
    print("PostHoc evaluation complete")
        
    for motif_idx, motif_id in enumerate(motif_list):
        print(f"Processing motif {motif_idx}")
        
        # Prepare the CSV row
        row = [
            motif_idx,
            motif_id,
            gnnex_weights[motif_idx].item(),
            pgex_weights[motif_idx].item(),
        ]
        
        csv_data.append(row)
        
    #Unknown motifs
    row = [
        -1,
        "UNK",
        gnnex_weights[len(motif_list)].item(),
        pgex_weights[len(motif_list)].item(),
    ]
    csv_data.append(row)
                
    # Determine the headers based on vanilla_model presence
    headers = [
        "motif_id",
        "motif",
        "gnnex_importance",
        "pgex_importance"
    ]

    # Write data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_data)
        
# def save_csv_motif_importance(model, motif_list, masked_data, csv_file_path, vanilla_model=False):
#     '''
#     Example usage save_csv_motif_importance(model, motif_list, [(train_mask_data, training_data)], train_file, vanilla_model= vanilla_model, lookup = test_data_lookup)
#     '''
    
#     # Get the device from the model
#     model_device = next(model.parameters()).device
#     csv_data = []  # Collect data for the CSV file
#     use_vanilla = vanilla_model
#     if not use_vanilla:
#         motif_weights = model.motif_params.detach().cpu()
    
        
#     # Process each dataset: train, val, test
#     for dataset in masked_data:
#         for motif_idx in dataset[0]:
#             print(f"Processing motif {motif_idx}")
#             logit_diff = torch.tensor([[0.0]], device=model_device)
#             log_probabilities_diff = torch.tensor([[0.0]], device=model_device)
#             for graph_idx in dataset[0][motif_idx]: #masked data list for each motif
#                 data = dataset[1][graph_idx].to(model_device) #original data list

#                 # Original and perturbed predictions using the main model
#                 original_pred, _ = model(data.x, data.edge_index, None, node_to_motifs=data.nodes_to_motifs)
#                 new_pred, _ = model(
#                     dataset[0][motif_idx][graph_idx].to(model_device),
#                     data.edge_index,
#                     None,
#                     node_to_motifs=data.nodes_to_motifs
#                 )
                
#                 logit_diff += original_pred - new_pred
                
                    
#                 if motif_idx == -1:
#                     # Prepare the CSV row
#                     row = [
#                         -1,
#                         "UNK",
#                         graph_idx,
#                         data.smiles,
#                         original_pred.item(),
#                         new_pred.item(),
#                         F.logsigmoid(original_pred).item(),
#                         F.logsigmoid(new_pred).item(),
#                     ]
#                     if not use_vanilla:
#                         row.extend([
#                             99,
#                             1.0,
#                         ])
#                 else:    
#                     # Prepare the CSV row
#                     row = [
#                         motif_idx,
#                         motif_list[motif_idx],
#                         graph_idx,
#                         data.smiles,
#                         original_pred.item(),
#                         new_pred.item(),
#                         F.logsigmoid(original_pred).item(),
#                         F.logsigmoid(new_pred).item(),
#                     ]
#                     if not use_vanilla:
#                         row.extend([
#                             motif_weights[motif_idx].item(),
#                             torch.sigmoid(motif_weights[motif_idx]).item(),
#                         ])
#                 row.append(data.y.item())
#                 csv_data.append(row)
    
#     # Determine the headers based on vanilla_model presence
#     headers = [
#         "motif_id",
#         "motif",
#         "graph_id",
#         "graph_str",
#         "original_logit",
#         "new_logit",
#         "original_log_prob",
#         "new_log_prob"
#     ]
#     if not use_vanilla:
#         headers.extend([
#             "importance",
#             "sigmoid_importance",
#         ])
#     headers.append("class_label")

#     # Write data to the CSV file
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(headers)
#         writer.writerows(csv_data)

def save_csv_motif_importance(model, motif_list, masked_data, csv_file_path, vanilla_model=False, batch_size=32):
    """
    Optimized version of save_csv_motif_importance to reduce memory usage.
    Computes the logit differences per motif across a dataset and saves to CSV.
    """
    model_device = next(model.parameters()).device
    csv_data = []
    use_vanilla = vanilla_model

    motif_weights = None
    if not use_vanilla and hasattr(model, 'motif_params'):
        motif_weights = model.motif_params.detach().cpu()

    for mask_data, original_data_list in masked_data:
        for motif_idx, graph_indices in mask_data.items():
            print(f"Processing motif {motif_idx}")

            for graph_idx in graph_indices:
                data = original_data_list[graph_idx].to(model_device)

                # Perturbed input data
                perturbed_data = mask_data[motif_idx][graph_idx].to(model_device)

                # Forward passes
                with torch.no_grad():
                    original_pred, _ = model(data.x, data.edge_index, None, node_to_motifs=data.nodes_to_motifs)
                    new_pred, _ = model(perturbed_data, data.edge_index, None, node_to_motifs=data.nodes_to_motifs)

                if motif_idx == -1:
                    motif_str = "UNK"
                    importance = 99
                    sigmoid_imp = 1.0
                else:
                    motif_str = motif_list[motif_idx]
                    importance = motif_weights[motif_idx].item() if motif_weights is not None else None
                    sigmoid_imp = torch.sigmoid(torch.tensor(importance)).item() if importance is not None else None

                row = [
                    motif_idx,
                    motif_str,
                    graph_idx,
                    data.smiles,
                    original_pred.item(),
                    new_pred.item(),
                    F.logsigmoid(original_pred).item(),
                    F.logsigmoid(new_pred).item()
                ]
                if motif_weights is not None:#if importance is not None and sigmoid_imp is not None:
                    row.extend([importance, sigmoid_imp])

                row.append(data.y.item())
                csv_data.append(row)

    headers = [
        "motif_id", "motif", "graph_id", "graph_str",
        "original_logit", "new_logit",
        "original_log_prob", "new_log_prob"
    ]
    if motif_weights is not None:
        headers.extend(["importance", "sigmoid_importance"])
    headers.append("class_label")

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_data)

        
def save_csv_motif_importance_multiclass(model, motif_list, masked_data, csv_file_path, num_classes, vanilla_model=False, batch_size=32):
    # Get the device from the model
    model_device = next(model.parameters()).device
    csv_data = []  # Collect data for the CSV file
    use_vanilla = vanilla_model is not None
    
    motif_weights = None
    if not use_vanilla and hasattr(model, 'motif_params'):
        motif_weights = model.motif_params.detach().cpu()

    for mask_data, original_data_list in masked_data:
        for motif_idx, graph_indices in mask_data.items():
            print(f"Processing motif {motif_idx}")
            for graph_idx in graph_indices:
                data = original_data_list[graph_idx].to(model_device)
                valid_mask = ~torch.isnan(data.y)
                
                # Skip graphs with all NaN labels
                if not valid_mask.any():
                    continue
                    
                # Perturbed input data
                perturbed_data = mask_data[motif_idx][graph_idx].to(model_device)

                # Forward passes
                with torch.no_grad():
                    original_pred, _ = model(data.x, data.edge_index, None, node_to_motifs=data.nodes_to_motifs)
                    new_pred, _ = model(perturbed_data, data.edge_index, None, node_to_motifs=data.nodes_to_motifs)
                    
                # Collect data for each class
                for class_idx in range(num_classes):
                    if valid_mask[:,class_idx].item():  # Check if the label for this class is valid
                        if motif_idx == -1:
                            motif_str = "UNK"
                            importance = 99
                            sigmoid_imp = 1.0
                        else:
                            motif_str = motif_list[motif_idx]
                            importance = motif_weights[motif_idx, class_idx].item()
                            sigmoid_imp = torch.sigmoid(motif_weights[motif_idx, class_idx]).item()

                        
                        
                        
                        row = [
                            motif_idx,
                            motif_id,
                            graph_idx,
                            data.smiles,
                            class_idx,
                            original_prediction[:, class_idx].item(),
                            new_prediction[:, class_idx].item(),
                            F.log_softmax(original_prediction[:, class_idx], dim=-1).item(),
                            F.log_softmax(new_prediction[:, class_idx], dim=-1).item(),
                        ]
                        if motif_weights is not None:
                            row.extend([
                                importance_class,
                                sigmoid_importance_class,
                            ])
                        row.append(float(data.y[:, class_idx].item()))
                        csv_data.append(row)


    # Determine the headers based on vanilla_model presence
    headers = [
        "motif_id",
        "motif",
        "graph_id",
        "graph_str",
        "class_id",
        "original_logit",
        "new_logit",
        "original_log_prob",
        "new_log_prob"
    ]
    if use_vanilla:
        headers.extend([
        "importance",
        "sigmoid_importance",
        ])
    headers.append("class_label")

    # Write data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_data)

# def save_csv_motif_importance(model, epoch, image_dir, motif_list, image_files, masked_data, csv_file_path):
#     # Get the device from the model
#     model_device = next(model.parameters()).device
#     motif_weights = model.motif_params.detach().cpu()
#     csv_data = []  # Collect data for the CSV file
    
#     for motif_idx, motif_id in enumerate(motif_list):
#         print(f"Processing motif {motif_idx}")
#         logit_diff = torch.tensor([[0.0]], device=model_device)
#         total_graphs = 0  # To track the total number of graphs across train, val, and test
        
#         # Process each dataset: train, val, test
#         for dataset_idx, dataset in enumerate(masked_data):
#             for graph_idx in dataset[0][motif_idx]:
#                 total_graphs += 1  # Count graphs
#                 data = dataset[1][graph_idx].to(model_device)
                
#                 # Original and perturbed predictions
#                 original_prediction, _ = model(data.x, data.edge_index, None, node_to_motifs = data.nodes_to_motifs)
#                 new_prediction, _ = model(
#                     dataset[0][motif_idx][graph_idx].to(model_device), 
#                     data.edge_index, 
#                     None, 
#                     node_to_motifs = data.nodes_to_motifs
#                 )
                
#                 logit_diff += original_prediction - new_prediction
                
#                 # Collect data for CSV
                
#                 importance_class_0 = motif_weights[motif_idx].item()
#                 sigmoid_importance_class_0 = torch.sigmoid(motif_weights[motif_idx]).item()
#                 csv_data.append([
#                     motif_idx,
#                     motif_id,
#                     graph_idx,
#                     importance_class_0,
#                     sigmoid_importance_class_0,
#                     original_prediction.item(),
#                     new_prediction.item(),
#                     data.y.item()  # data.y` contains the class label
#                 ])
        
#     # Write data to the CSV file
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             "motif_id",
#             "motif",
#             "graph_id", 
#             "importance", 
#             "sigmoid_importance", 
#             "original_logit", 
#             "new_logit", 
#             "class_label"
#         ])
#         writer.writerows(csv_data)
        
# def save_csv_motif_importance_with_vanilla(model, vanilla_model, epoch, image_dir, motif_list, image_files, masked_data, csv_file_path):
#     # Get the device from the model
#     model_device = next(model.parameters()).device
#     motif_weights = model.motif_params.detach().cpu()
#     csv_data = []  # Collect data for the CSV file
    
#     for motif_idx, motif_id in enumerate(motif_list):
#         print(f"Processing motif {motif_idx}")
#         logit_diff = torch.tensor([[0.0]], device=model_device)
#         logit_diff_with_vanilla = torch.tensor([[0.0]], device=model_device)
#         total_graphs = 0  # To track the total number of graphs across train, val, and test
        
#         # Process each dataset: train, val, test
#         for dataset_idx, dataset in enumerate(masked_data):
#             for graph_idx in dataset[0][motif_idx]:
#                 total_graphs += 1  # Count graphs
#                 data = dataset[1][graph_idx].to(model_device)
                
#                 # Original and perturbed predictions
#                 original_prediction, _ = model(data.x, data.edge_index, None, node_to_motifs = data.nodes_to_motifs)
#                 new_prediction, _ = model(
#                     dataset[0][motif_idx][graph_idx].to(model_device), 
#                     data.edge_index, 
#                     None, 
#                     node_to_motifs = data.nodes_to_motifs
#                 )
                
#                 logit_diff += original_prediction - new_prediction
                
#                 original_prediction_with_vanilla, _ =vanilla_model(data.x, data.edge_index, None, node_to_motifs = data.nodes_to_motifs)
#                 new_prediction_with_vanilla, _ = vanilla_model(dataset[0][motif_idx][graph_idx].to(model_device), 
#                                                         data.edge_index, 
#                                                         None, 
#                                                         node_to_motifs = data.nodes_to_motifs)
#                 logit_diff_with_vanilla += original_prediction_with_vanilla - new_prediction_with_vanilla
                
#                 # Collect data for CSV
                
#                 importance_class_0 = motif_weights[motif_idx].item()
#                 sigmoid_importance_class_0 = torch.sigmoid(motif_weights[motif_idx]).item()
#                 csv_data.append([
#                     motif_idx,
#                     motif_id,
#                     graph_idx,
#                     importance_class_0,
#                     sigmoid_importance_class_0,
#                     original_prediction.item(),
#                     new_prediction.item(),
#                     original_prediction_with_vanilla.item(),
#                     new_prediction_with_vanilla.item(),
#                     data.y.item()  # data.y` contains the class label
#                 ])
        
#     # Write data to the CSV file
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             "motif_id",
#             "motif",
#             "graph_id", 
#             "importance", 
#             "sigmoid_importance", 
#             "original_logit", 
#             "new_logit", 
#             "original_logit_vanilla", 
#             "new_logit_vanilla", 
#             "class_label"
#         ])
#         writer.writerows(csv_data)

        

        
# def save_csv_motif_importance_multiclass_with_vanilla(model, vanilla_model, epoch, image_dir, motif_list, image_files, masked_data, csv_file_path):
#     # Get the device from the model
#     model_device = next(model.parameters()).device
#     num_classes = model.motif_params.shape[1]  # Determine the number of classes dynamically
#     motif_weights = model.motif_params.detach().cpu()
#     csv_data = []  # Collect data for the CSV file

#     for motif_idx, motif_id in enumerate(motif_list):
#         print(f"Processing motif {motif_idx}")
#         logit_diff = None
#         total_graphs = 0  # To track the total number of graphs across datasets

#         # Process each dataset in masked_data
#         for dataset_idx, dataset in enumerate(masked_data):
#             for graph_idx in dataset[0][motif_idx]:
#                 data = dataset[1][graph_idx].to(model_device)

#                 valid_mask = ~torch.isnan(data.y)
                
#                 # Skip graphs with all NaN labels
#                 if not valid_mask.any():
#                     continue

#                 total_graphs += 1  # Count graphs with valid labels

#                 # Original and perturbed predictions
#                 original_prediction, _ = model(data.x, data.edge_index, None, data.smiles)
#                 new_prediction, _ = model(
#                     dataset[0][motif_idx][graph_idx].to(model_device),
#                     data.edge_index,
#                     None,
#                     data.smiles
#                 )
#                 if logit_diff is None:
#                     logit_diff = torch.zeros_like(original_prediction, device=model_device)
#                     logit_diff_with_vanilla = torch.zeros_like(original_prediction, device=model_device)
                

#                 # Accumulate logit differences for all classes
#                 logit_diff += (original_prediction - new_prediction) * valid_mask.float()
                
#                 original_prediction_with_vanilla, _ =vanilla_model(data.x, data.edge_index, None, data.smiles)
#                 new_prediction_with_vanilla, _ = vanilla_model(dataset[0][motif_idx][graph_idx].to(model_device), 
#                                                         data.edge_index, 
#                                                         None, 
#                                                         data.smiles)
#                 logit_diff_with_vanilla += original_prediction_with_vanilla - new_prediction_with_vanilla

#                 # Collect data for each class
#                 for class_idx in range(num_classes):
#                     if valid_mask[:,class_idx].item():  # Check if the label for this class is valid
#                         importance_class = motif_weights[motif_idx, class_idx].item()
#                         sigmoid_importance_class = torch.sigmoid(motif_weights[motif_idx, class_idx]).item()
#                         csv_data.append([
#                             motif_idx,
#                             motif_id,
#                             graph_idx,
#                             class_idx,
#                             importance_class,
#                             sigmoid_importance_class,
#                             original_prediction[:,class_idx].item(),
#                             new_prediction[:,class_idx].item(),
#                             original_prediction_with_vanilla[:,class_idx].item(),
#                             new_prediction_with_vanilla[:,class_idx].item(),
#                             float(data.y[:,class_idx].item())
#                         ])

#     # Write data to the CSV file
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             "motif_id",
#             "motif",
#             "graph_id",
#             "class_id",
#             "importance",
#             "sigmoid_importance",
#             "original_logit",
#             "new_logit",
#             "original_logit_vanilla", 
#             "new_logit_vanilla", 
#             "class_label"
#         ])
#         writer.writerows(csv_data)
