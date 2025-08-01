#!/bin/bash

#SBATCH --time=2-00:00:00 --mem=200G

source ~/hpc-share/anaconda3/etc/profile.d/conda.sh
conda activate l2xgnn


python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 0 --algorithm RBRICS --layer_type GAT --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.0001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-0-GAT-EXPLLR0.01-GNNLR0.0001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 0 --algorithm RBRICS --layer_type GIN --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.0001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-0-GIN-EXPLLR0.01-GNNLR0.0001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 0 --algorithm RBRICS --layer_type SAGE --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.0001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-0-SAGE-EXPLLR0.01-GNNLR0.0001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 1 --algorithm RBRICS --layer_type GAT --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.0001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-1-GAT-EXPLLR0.01-GNNLR0.0001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 1 --algorithm RBRICS --layer_type GIN --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.0001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-1-GIN-EXPLLR0.01-GNNLR0.0001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 1 --algorithm RBRICS --layer_type SAGE --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.0001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-1-SAGE-EXPLLR0.01-GNNLR0.0001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 1 --algorithm RBRICS --layer_type GCN --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.0001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-1-GCN-EXPLLR0.01-GNNLR0.0001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 2 --algorithm RBRICS --layer_type GCN --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.0001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-2-GCN-EXPLLR0.01-GNNLR0.0001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 0 --algorithm RBRICS --layer_type GAT --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-0-GAT-EXPLLR0.01-GNNLR0.001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 0 --algorithm RBRICS --layer_type SAGE --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-0-SAGE-EXPLLR0.01-GNNLR0.001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 0 --algorithm RBRICS --layer_type GCN --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-0-GCN-EXPLLR0.01-GNNLR0.001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 1 --algorithm RBRICS --layer_type GAT --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-1-GAT-EXPLLR0.01-GNNLR0.001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 1 --algorithm RBRICS --layer_type GIN --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-1-GIN-EXPLLR0.01-GNNLR0.001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 1 --algorithm RBRICS --layer_type SAGE --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-1-SAGE-EXPLLR0.01-GNNLR0.001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 1 --algorithm RBRICS --layer_type GCN --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-1-GCN-EXPLLR0.01-GNNLR0.001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10

python 1_run_EXPT_SingleChannel.py --dataset_name Lipophilicity --seed 0 --fold 2 --algorithm RBRICS --layer_type GAT --task_type Regression --model_type SingleChannel --num_mp_layers 2 --hidden 32 --expl_lr 0.01 --lr 0.001 --date_tag RBRICS0.5 --epochs 500 --output_dir /nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/RBRICS_MINORITY_MP2_DIM32/MOSE_Reg/EXPT-14R-Lipophilicity-SEED-0-FOLD-2-GAT-EXPLLR0.01-GNNLR0.001-SingleChannel-RBRICS
pkill -f 1_run_EXPT_SingleChannel.py
sleep 10
