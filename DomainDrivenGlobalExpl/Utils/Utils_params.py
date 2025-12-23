import torch
import numpy as np
import sys
from collections import defaultdict
from torch_geometric.loader import DataLoader
from Utils.Utils_Train import get_masked_graphs_from_list
import pdb
import csv
from Explainer.Explainer import Explainer
from Explainer.GNNExplainer import GNNExplainer
from Explainer.PGExplainer import PGExplainer
from tqdm import tqdm
from scipy.stats import pearsonr
import torch.nn.functional as F
import wandb
import pandas as pd

from torch_geometric.data import Batch

def _sigmoid_np(x):
    return 1 / (1 + np.exp(-x))

def save_posthoc_motif_importance(model, motif_list, masked_data, csv_file_path, lookup=None, test_lookup=None, task_type="BinaryClass", dataset_name = None, model_type = None):
    '''
    Example usage save_csv_motif_importance(model, motif_list, [(train_mask_data, training_data)], train_file, vanilla_model= vanilla_model, lookup = test_data_lookup)
    '''
    model.eval()
    
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




def save_csv_motif_importance_optimized(
    model,
    motif_list,
    masked_data,
    csv_file_path,
    vanilla_model=False,
    batch_size=32,
    num_classes=1,   # NEW
):
    model.eval()
    model_device = next(model.parameters()).device
    csv_data = []
    use_vanilla = vanilla_model

    motif_weights = None
    if not use_vanilla and hasattr(model, 'motif_params'):
        motif_weights = model.motif_params.detach().cpu()

    # ----------------- Small helpers -----------------

    def _selected_indices(graph_indices):
        """Normalize indices to a sorted unique list."""
        if isinstance(graph_indices, (list, tuple, torch.Tensor)):
            return sorted(set(int(i) for i in graph_indices))
        else:
            return sorted(set(int(i) for i in list(graph_indices)))

    def _log_pearson_metrics_from_csv(csv_data, motif_weights, headers):
        """
        Shared Pearson / weighted Pearson logging for both binary and multiclass.
        Uses headers to locate the relevant columns.
        """
        has_imp = motif_weights is not None
        if not (has_imp and csv_data):
            wandb.log({
                "pearson_motif_logit_impact": 0.0,
                "weighted_pearson_motif_logit_impact": 0.0,
                "zero_ratio_penalty": 0.0,
                "n_motifs_used": 0,
            })
            return

        try:
            col_motif_id   = headers.index("motif_id")
            col_orig_logit = headers.index("original_logit")
            col_new_logit  = headers.index("new_logit")
            col_sigimp     = headers.index("sigmoid_importance")
        except ValueError:
            # Missing columns (e.g., no sigmoid_importance) → log zeros
            wandb.log({
                "pearson_motif_logit_impact": 0.0,
                "weighted_pearson_motif_logit_impact": 0.0,
                "zero_ratio_penalty": 0.0,
                "n_motifs_used": 0,
            })
            return

        arr = np.asarray(csv_data, dtype=object)

        # Filter: motif_id != -1 (exclude UNK)
        mask = arr[:, col_motif_id] != -1
        if not np.any(mask):
            wandb.log({
                "pearson_motif_logit_impact": 0.0,
                "weighted_pearson_motif_logit_impact": 0.0,
                "zero_ratio_penalty": 0.0,
                "n_motifs_used": 0,
            })
            return

        orig = arr[mask, col_orig_logit].astype(np.float64, copy=False)
        new  = arr[mask, col_new_logit].astype(np.float64, copy=False)
        sigi = arr[mask, col_sigimp].astype(np.float64, copy=False)

        # Drop NaNs / inf in a single pass
        finite_mask = np.isfinite(orig) & np.isfinite(new) & np.isfinite(sigi)
        if not np.any(finite_mask):
            wandb.log({
                "pearson_motif_logit_impact": 0.0,
                "weighted_pearson_motif_logit_impact": 0.0,
                "zero_ratio_penalty": 0.0,
                "n_motifs_used": 0,
            })
            return

        orig = orig[finite_mask]
        new  = new[finite_mask]
        sigi = sigi[finite_mask]

        # Compute impact = |sigmoid(orig) - sigmoid(new)| (vectorized)
        impact = np.abs(_sigmoid_np(orig) - _sigmoid_np(new))

        # Guard against constant arrays (variance == 0) without counting tiny fp noise
        if impact.size > 1 and sigi.size > 1 and np.var(impact) > 0.0 and np.var(sigi) > 0.0:
            x = sigi
            y = impact
            x_mean = x.mean()
            y_mean = y.mean()
            xm = x - x_mean
            ym = y - y_mean
            denom = (np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum()))
            pearson_corr = float((xm * ym).sum() / denom) if denom > 0 else 0.0

            zero_ratio = float((sigi < 0.1).mean())
            penalty = float(1.0 - zero_ratio)
            weighted_pearson = float(pearson_corr * penalty)
        else:
            pearson_corr = 0.0
            weighted_pearson = 0.0
            penalty = 0.0

        wandb.log({
            "pearson_motif_logit_impact": pearson_corr,
            "weighted_pearson_motif_logit_impact": weighted_pearson,
            "zero_ratio_penalty": penalty,
            "n_motifs_used": int(impact.size),
        })

    # ==========================================================
    # 1) BINARY CASE: original behavior kept intact (num_classes = 1 or 2)
    # ==========================================================
    if num_classes <= 2:
        with torch.no_grad():
            for mask_data, original_data_list in masked_data:
                # ---------- Full batch once for original logits ----------
                full_batch: Batch = Batch.from_data_list(original_data_list).to(model_device)
                full_batch_vec = getattr(full_batch, "batch", None)
                full_nodes_to_motifs = getattr(full_batch, "nodes_to_motifs", None)
                
                orig_logits, _ = model(
                    full_batch.x,
                    full_batch.edge_index,
                    full_batch_vec,
                    node_to_motifs=full_nodes_to_motifs
                )  # [num_graphs]
                orig_logits = orig_logits.view(-1).detach().cpu()
                orig_logsig = F.logsigmoid(orig_logits)

                # Per-graph metadata (CPU)
                smiles_list = [d.smiles for d in original_data_list]
                y_vec = torch.tensor(
                    [int(d.y.item()) for d in original_data_list],
                    dtype=torch.long
                )

                # Iterate only the motifs we care about
                for motif_idx, graph_indices in mask_data.items():
                    print(f"Processing motif {motif_idx}")

                    selected_indices = _selected_indices(graph_indices)
                    if len(selected_indices) == 0:
                        continue

                    # ---------- Build sub-batch only for the selected graphs ----------
                    sub_list = [original_data_list[i] for i in selected_indices]
                    sub_batch: Batch = Batch.from_data_list(sub_list).to(model_device)
                    sub_batch_vec = getattr(sub_batch, "batch", None)
                    sub_nodes_to_motifs = getattr(sub_batch, "nodes_to_motifs", None)

                    # ---------- One masked forward for this motif over the subset ----------
                    masked_logits_sub, _ = model(
                        sub_batch.x,
                        sub_batch.edge_index,
                        sub_batch_vec,
                        node_to_motifs=sub_nodes_to_motifs,
                        masked_motif=motif_idx
                    )
                    masked_logits_sub = masked_logits_sub.view(-1).detach().cpu()
                    masked_logsig_sub = F.logsigmoid(masked_logits_sub)

                    # Motif-level metadata
                    if motif_idx == -1:
                        motif_str = "UNK"
                        if hasattr(model, "unk_param"):
                            importance = float(model.unk_param.item())
                            sigmoid_imp = float(model.unk_param.sigmoid().item())
                        else:
                            importance = 99.0
                            if hasattr(model, "unk_importance"):
                                sigmoid_imp = float(model.unk_importance)
                            else:
                                sigmoid_imp = None
                    else:
                        motif_str = motif_list[motif_idx]
                        if motif_weights is not None:
                            importance = float(motif_weights[motif_idx].item())
                            sigmoid_imp = float(
                                torch.sigmoid(torch.as_tensor(importance)).item()
                            )
                        else:
                            importance = None
                            sigmoid_imp = None

                    # ---------- Emit rows only for the selected graphs ----------
                    for k, gidx in enumerate(selected_indices):
                        row = [
                            int(motif_idx),
                            motif_str,
                            int(gidx),
                            smiles_list[gidx],
                            float(orig_logits[gidx].item()),
                            float(masked_logits_sub[k].item()),
                            float(orig_logsig[gidx].item()),
                            float(masked_logsig_sub[k].item()),
                        ]
                        if motif_weights is not None:
                            row.extend([importance, sigmoid_imp])
                        row.append(int(y_vec[gidx].item()))
                        csv_data.append(row)
    
        # ---- Build headers (unchanged for binary) ----
        headers = [
            "motif_id", "motif", "graph_id", "graph_str",
            "original_logit", "new_logit",
            "original_log_prob", "new_log_prob"
        ]
        if motif_weights is not None:
            headers.extend(["importance", "sigmoid_importance"])
        headers.append("class_label")

        # ---- Vectorized metrics (shared helper) ----
        _log_pearson_metrics_from_csv(csv_data, motif_weights, headers)

        # ---- Write CSV (fast, streaming; avoids DataFrame build) ----
        with open(csv_file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)
        return  # done with binary case

    # ==========================================================
    # 2) MULTICLASS / MULTILABEL CASE (num_classes > 1)
    # ==========================================================
    with torch.no_grad():
        for mask_data, original_data_list in masked_data:
            # ---------- Full batch once for original logits ----------
            full_batch: Batch = Batch.from_data_list(original_data_list).to(model_device)
            full_batch_vec = getattr(full_batch, "batch", None)
            full_nodes_to_motifs = getattr(full_batch, "nodes_to_motifs", None)
            
            orig_logits_raw, _ = model(
                full_batch.x,
                full_batch.edge_index,
                full_batch_vec,
                node_to_motifs=full_nodes_to_motifs
            )  # [num_graphs, num_classes] expected

            orig_logits_raw = orig_logits_raw.view(-1, num_classes)
            orig_logprob_raw = F.log_softmax(orig_logits_raw, dim=-1)

            orig_logits = orig_logits_raw.detach().cpu()      # [G, C]
            orig_logprob = orig_logprob_raw.detach().cpu()    # [G, C]

            # Per-graph metadata (CPU)
            smiles_list = [d.smiles for d in original_data_list]

            # Build label matrix [G, C], with NaNs for missing labels
            y_rows = []
            for d in original_data_list:
                y_flat = d.y.view(-1).to(torch.float32).cpu()
                if y_flat.numel() == num_classes:
                    pass
                elif y_flat.numel() == 1:
                    # broadcast scalar label if needed
                    y_flat = y_flat.repeat(num_classes)
                elif y_flat.numel() < num_classes:
                    pad = torch.full((num_classes - y_flat.numel(),), float('nan'))
                    y_flat = torch.cat([y_flat, pad], dim=0)
                else:
                    y_flat = y_flat[:num_classes]
                y_rows.append(y_flat)
            y_mat = torch.stack(y_rows, dim=0)          # [G, C]
            valid_mask = ~torch.isnan(y_mat)            # [G, C] boolean
            num_graphs = len(original_data_list)

            # Iterate only the motifs we care about
            for motif_idx, graph_indices in mask_data.items():
                print(f"Processing motif {motif_idx}")

                selected_indices = _selected_indices(graph_indices)
                if len(selected_indices) == 0:
                    continue

                # ---------- Build sub-batch only for the selected graphs ----------
                sub_list = [original_data_list[i] for i in selected_indices]
                sub_batch: Batch = Batch.from_data_list(sub_list).to(model_device)
                sub_batch_vec = getattr(sub_batch, "batch", None)
                sub_nodes_to_motifs = getattr(sub_batch, "nodes_to_motifs", None)

                # ---------- One masked forward for this motif over the subset ----------
                masked_logits_raw, _ = model(
                    sub_batch.x,
                    sub_batch.edge_index,
                    sub_batch_vec,
                    node_to_motifs=sub_nodes_to_motifs,
                    masked_motif=motif_idx
                )
                masked_logits_raw = masked_logits_raw.view(-1, num_classes)
                masked_logprob_raw = F.log_softmax(masked_logits_raw, dim=-1)

                masked_logits_sub = masked_logits_raw.detach().cpu()   # [G_sub, C]
                masked_logprob_sub = masked_logprob_raw.detach().cpu() # [G_sub, C]

                # Motif-level metadata (string only; importance is per-class below)
                if motif_idx == -1:
                    motif_str = "UNK"
                else:
                    motif_str = motif_list[motif_idx]

                # ---------- Emit rows for each graph & class ----------
                for k, gidx in enumerate(selected_indices):
                    if gidx < 0 or gidx >= num_graphs:
                        continue

                    for class_idx in range(num_classes):
                        if not bool(valid_mask[gidx, class_idx].item()):
                            continue  # skip unlabeled class

                        # importance & sigmoid importance per (motif, class)
                        if motif_idx == -1:
                            if hasattr(model, "unk_param"):
                                unk_param = model.unk_param.detach().cpu()
                                if unk_param.ndim == 0:
                                    importance = float(unk_param.item())
                                elif unk_param.ndim == 1 and unk_param.shape[0] == num_classes:
                                    importance = float(unk_param[class_idx].item())
                                else:
                                    importance = float(unk_param.view(-1)[0].item())
                                sigmoid_imp = float(
                                    torch.sigmoid(torch.as_tensor(importance)).item()
                                )
                            else:
                                importance = 99.0
                                if hasattr(model, "unk_importance"):
                                    sigmoid_imp = float(model.unk_importance)
                                else:
                                    sigmoid_imp = None
                        else:
                            if motif_weights is not None:
                                if motif_weights.ndim == 1:
                                    importance = float(motif_weights[motif_idx].item())
                                elif motif_weights.ndim == 2:
                                    importance = float(motif_weights[motif_idx, class_idx].item())
                                else:
                                    importance = float(motif_weights[motif_idx].view(-1)[0].item())
                                sigmoid_imp = float(
                                    torch.sigmoid(torch.as_tensor(importance)).item()
                                )
                            else:
                                importance = None
                                sigmoid_imp = None

                        row = [
                            int(motif_idx),
                            motif_str,
                            int(gidx),
                            smiles_list[gidx],
                            int(class_idx),
                            float(orig_logits[gidx, class_idx].item()),
                            float(masked_logits_sub[k, class_idx].item()),
                            float(orig_logprob[gidx, class_idx].item()),
                            float(masked_logprob_sub[k, class_idx].item()),
                        ]
                        if motif_weights is not None or motif_idx == -1:
                            row.extend([importance, sigmoid_imp])
                        row.append(float(y_mat[gidx, class_idx].item()))
                        csv_data.append(row)

    # ---- Build headers for MULTICLASS ----
    headers = [
        "motif_id", "motif", "graph_id", "graph_str",
        "class_id",
        "original_logit", "new_logit",
        "original_log_prob", "new_log_prob"
    ]
    if motif_weights is not None or any(r[0] == -1 for r in csv_data):
        headers.extend(["importance", "sigmoid_importance"])
    headers.append("class_label")

    # ---- Vectorized metrics for MULTICLASS (shared helper) ----
    _log_pearson_metrics_from_csv(csv_data, motif_weights, headers)

    # ---- Write CSV (multiclass) ----
    with open(csv_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_data)


        

# def save_csv_motif_importance_optimized(model, motif_list, masked_data, csv_file_path, vanilla_model=False, batch_size=32):
#     model.eval()
#     model_device = next(model.parameters()).device
#     csv_data = []
#     use_vanilla = vanilla_model

#     motif_weights = None
#     if not use_vanilla and hasattr(model, 'motif_params'):
#         motif_weights = model.motif_params.detach().cpu()
        
#     with torch.no_grad():
#         for mask_data, original_data_list in masked_data:
#             # ---------- Full batch once for original logits ----------
#             full_batch: Batch = Batch.from_data_list(original_data_list).to(model_device)
#             full_batch_vec = getattr(full_batch, "batch", None)
#             full_nodes_to_motifs = getattr(full_batch, "nodes_to_motifs", None)
            
#             orig_logits, _ = model(
#                 full_batch.x,
#                 full_batch.edge_index,
#                 full_batch_vec,
#                 node_to_motifs=full_nodes_to_motifs
#             )  # [num_graphs]
#             orig_logits = orig_logits.view(-1).detach().cpu()
#             orig_logsig = F.logsigmoid(orig_logits)

#             # Per-graph metadata (CPU)
#             smiles_list = [d.smiles for d in original_data_list]
#             y_vec = torch.tensor([int(d.y.item()) for d in original_data_list], dtype=torch.long)
#             num_graphs = len(original_data_list)

#             # Iterate only the motifs we care about
#             for motif_idx, graph_indices in mask_data.items():
#                 print(f"Processing motif {motif_idx}")
#                 # Normalize indices to a sorted unique list (in case input is set/dup)
#                 if isinstance(graph_indices, (list, tuple, torch.Tensor)):
#                     selected_indices = sorted(set(int(i) for i in graph_indices))
#                 else:
#                     # If mask_data[motif_idx] is some custom container, coerce it
#                     selected_indices = sorted(set(int(i) for i in list(graph_indices)))

#                 # If this motif applies to no graphs, skip the masked forward entirely
#                 if len(selected_indices) == 0:
#                     continue

#                 # ---------- Build sub-batch only for the selected graphs ----------
#                 sub_list = [original_data_list[i] for i in selected_indices]
#                 sub_batch: Batch = Batch.from_data_list(sub_list).to(model_device)
#                 sub_batch_vec = getattr(sub_batch, "batch", None)
#                 sub_nodes_to_motifs = getattr(sub_batch, "nodes_to_motifs", None)

#                 # ---------- One masked forward for this motif over the subset ----------
#                 masked_logits_sub, _ = model(
#                     sub_batch.x,
#                     sub_batch.edge_index,
#                     sub_batch_vec,
#                     node_to_motifs=sub_nodes_to_motifs,
#                     masked_motif=motif_idx
#                 )
#                 masked_logits_sub = masked_logits_sub.view(-1).detach().cpu()
#                 masked_logsig_sub = F.logsigmoid(masked_logits_sub)

#                 # Motif-level metadata
#                 if motif_idx == -1:
#                     motif_str = "UNK"
#                     if hasattr(model, "unk_param"):
#                         importance = float(model.unk_param.item())
#                         sigmoid_imp = float(model.unk_param.sigmoid().item())
#                     else:
#                         importance = 99.0
#                         if hasattr(model, "unk_importance"):
#                             sigmoid_imp = float(model.unk_importance)
#                         else:
#                             sigmoid_imp = None
#                 else:
#                     motif_str = motif_list[motif_idx]
#                     if motif_weights is not None:
#                         importance = float(motif_weights[motif_idx].item())
#                         sigmoid_imp = float(torch.sigmoid(torch.as_tensor(importance)).item())
#                     else:
#                         importance = None
#                         sigmoid_imp = None

#                 # ---------- Emit rows only for the selected graphs ----------
#                 for k, gidx in enumerate(selected_indices):
#                     row = [
#                         int(motif_idx),
#                         motif_str,
#                         int(gidx),
#                         smiles_list[gidx],
#                         float(orig_logits[gidx].item()),
#                         float(masked_logits_sub[k].item()),
#                         float(orig_logsig[gidx].item()),
#                         float(masked_logsig_sub[k].item()),
#                     ]
#                     if motif_weights is not None:
#                         row.extend([importance, sigmoid_imp])
#                     row.append(int(y_vec[gidx].item()))
#                     csv_data.append(row)
             
    
#     # ---- Build headers (unchanged) ----
#     headers = [
#         "motif_id", "motif", "graph_id", "graph_str",
#         "original_logit", "new_logit",
#         "original_log_prob", "new_log_prob"
#     ]
#     has_imp = motif_weights is not None
#     if has_imp:
#         headers.extend(["importance", "sigmoid_importance"])
#     headers.append("class_label")

#     # ---- Vectorized metrics (no pandas) ----
#     if has_imp and csv_data:
#         # Convert once; keep as object, then slice/cast only what we need.
#         arr = np.asarray(csv_data, dtype=object)

#         # Column indices (matches headers above)
#         COL_MOTIF_ID   = 0
#         COL_ORIG_LOGIT = 4
#         COL_NEW_LOGIT  = 5
#         # importance = 8 (unused for correlation)
#         COL_SIGIMP     = 9  # "sigmoid_importance"

#         # Filter: motif_id != -1 (exclude UNK)
#         mask = arr[:, COL_MOTIF_ID] != -1
#         if np.any(mask):
#             # Pull needed columns and cast once
#             orig = arr[mask, COL_ORIG_LOGIT].astype(np.float64, copy=False)
#             new  = arr[mask, COL_NEW_LOGIT].astype(np.float64, copy=False)
#             sigi = arr[mask, COL_SIGIMP].astype(np.float64, copy=False)

#             # Drop NaNs / inf in a single pass
#             finite_mask = np.isfinite(orig) & np.isfinite(new) & np.isfinite(sigi)
#             if np.any(finite_mask):
#                 orig = orig[finite_mask]
#                 new  = new[finite_mask]
#                 sigi = sigi[finite_mask]

#                 # Compute impact = |sigmoid(orig) - sigmoid(new)| (vectorized)
#                 impact = np.abs(_sigmoid_np(orig) - _sigmoid_np(new))

#                 # Guard against constant arrays (variance == 0) without counting tiny fp noise
#                 if impact.size > 1 and sigi.size > 1 and np.var(impact) > 0.0 and np.var(sigi) > 0.0:
#                     # Pearson without scipy: fast and allocates 2 small scalars
#                     # corr = cov(x,y)/(std(x)*std(y))
#                     x = sigi
#                     y = impact
#                     x_mean = x.mean()
#                     y_mean = y.mean()
#                     xm = x - x_mean
#                     ym = y - y_mean
#                     denom = (np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum()))
#                     pearson_corr = float((xm * ym).sum() / denom) if denom > 0 else 0.0

#                     zero_ratio = float((sigi < 0.1).mean())
#                     penalty = float(1.0 - zero_ratio)
#                     weighted_pearson = float(pearson_corr * penalty)
#                 else:
#                     pearson_corr = 0.0
#                     weighted_pearson = 0.0
#                     penalty = 0.0

#                 # Log only once, with plain Python floats
#                 wandb.log({
#                     "pearson_motif_logit_impact": pearson_corr,
#                     "weighted_pearson_motif_logit_impact": weighted_pearson,
#                     "zero_ratio_penalty": penalty,
#                     "n_motifs_used": int(impact.size),
#                 })
#             else:
#                 wandb.log({
#                     "pearson_motif_logit_impact": 0.0,
#                     "weighted_pearson_motif_logit_impact": 0.0,
#                     "zero_ratio_penalty": 0.0,
#                     "n_motifs_used": 0,
#                 })
#         else:
#             wandb.log({
#                 "pearson_motif_logit_impact": 0.0,
#                 "weighted_pearson_motif_logit_impact": 0.0,
#                 "zero_ratio_penalty": 0.0,
#                 "n_motifs_used": 0,
#             })

#     # ---- Write CSV (fast, streaming; avoids DataFrame build) ----
#     with open(csv_file_path, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(headers)
#         writer.writerows(csv_data)
        
        

        

def save_csv_motif_importance(model, motif_list, masked_data, csv_file_path, vanilla_model=False, batch_size=32):
    """
    Optimized version of save_csv_motif_importance to reduce memory usage.
    Computes the logit differences per motif across a dataset and saves to CSV.
    """
    model.eval()
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
                    # new_pred, _ = model(perturbed_data, data.edge_index, None, node_to_motifs=data.nodes_to_motifs)
                    new_pred, _ = model(data.x, data.edge_index, None, node_to_motifs=data.nodes_to_motifs, masked_motif = motif_idx)

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
    
    if motif_weights is not None and csv_data:
        # Convert to DataFrame
        df = pd.DataFrame(csv_data, columns=headers)

        # Filter out UNK motifs (motif_idx == -1)
        df = df[df["motif_id"] != -1]

        # Compute impact: change in sigmoid(logit)
        df["impact"] = np.abs(_sigmoid_np(df["original_logit"].astype(float)) - _sigmoid_np(df["new_logit"].astype(float)))

        # Filter again in case of NaNs
        df = df.dropna(subset=["sigmoid_importance", "impact"])

        if df["sigmoid_importance"].nunique() > 1 and df["impact"].nunique() > 1:
            pearson_corr, _ = pearsonr(df["sigmoid_importance"], df["impact"])
            zero_ratio = np.mean(df["sigmoid_importance"] < 0.1)
            penalty = 1 - zero_ratio
            weighted_pearson = pearson_corr * penalty
        else:
            pearson_corr = 0.0
            weighted_pearson = 0.0
            penalty = 0.0

        wandb.log({
            "pearson_motif_logit_impact": pearson_corr,
            "weighted_pearson_motif_logit_impact": weighted_pearson,
            "zero_ratio_penalty": penalty,
            "n_motifs_used": len(df)
        })

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_data)

        
def save_csv_motif_importance_multiclass(model, motif_list, masked_data, csv_file_path, num_classes, vanilla_model=False, batch_size=32):
    # Get the device from the model
    model_device = next(model.parameters()).device
    csv_data = []  # Collect data for the CSV file
    use_vanilla = vanilla_model
    
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
                            importance = motif_weights[motif_idx, class_idx].item() if motif_weights is not None else None
                            sigmoid_imp = torch.sigmoid(motif_weights[motif_idx, class_idx]).item() if motif_weights is not None else None

                        
                        
                        
                        row = [
                            motif_idx,
                            motif_str,
                            graph_idx,
                            data.smiles,
                            class_idx,
                            original_pred[:, class_idx].item(),
                            new_pred[:, class_idx].item(),
                            F.log_softmax(original_pred[:, class_idx], dim=-1).item(),
                            F.log_softmax(new_pred[:, class_idx], dim=-1).item(),
                        ]
                        if motif_weights is not None:
                            row.extend([
                                importance,
                                sigmoid_imp,
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
    if motif_weights is not None:
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
