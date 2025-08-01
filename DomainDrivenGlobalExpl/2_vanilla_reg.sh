#!/bin/bash

#SBATCH --time=2-00:00:00 --mem=120G

# Initialize Conda
source ~/hpc-share/anaconda3/etc/profile.d/conda.sh

# Activate the desired environment
conda activate l2xgnn

for SEED in 0 
do
    for FOLD in 0 1 2 3 4
    do
        for ALGORITHM in RBRICS 
        do
        
            for LAYERTYPE in GCN GAT GIN SAGE
            do
            
                for MP_LAYERS in 2 3
                do
                    for HIDDEN_DIM in 16 32
                    do
                    
                        for GNN_LR in 0.001 0.0001
                        do
                        
                            for DATASETNAME in Lipophilicity
                            do

                                VanillaFOLDER="/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/Vanilla_Reg/EXPT-14R-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-${LAYERTYPE}-MP${MP_LAYERS}-DIM${HIDDEN_DIM}-GNNLR${GNN_LR}-Vanilla-None"
                                python 1_run_EXPT_SingleChannel.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type Regression --model_type Vanilla --num_mp_layers $MP_LAYERS --hidden $HIDDEN_DIM --lr $GNN_LR --date_tag RBRICS0.5 --epochs 500 --output_dir $VanillaFOLDER
                                # Kill any lingering Python processes
                                pkill -f 1_run_EXPT_SingleChannel.py
                                # Pause to allow memory reclamation
                                sleep 10

                            done

                            for DATASETNAME in esol
                            do

                                VanillaFOLDER="/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/STABLE/Vanilla_Reg/EXPT-14R-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-${LAYERTYPE}-MP${MP_LAYERS}-DIM${HIDDEN_DIM}-GNNLR${GNN_LR}-Vanilla-None"
                                python 1_run_EXPT_SingleChannel.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type Regression --model_type Vanilla --num_mp_layers $MP_LAYERS --hidden $HIDDEN_DIM --lr $GNN_LR --date_tag RBRICS0.2 --epochs 500 --output_dir $VanillaFOLDER
                                # Kill any lingering Python processes
                                pkill -f 1_run_EXPT_SingleChannel.py
                                # Pause to allow memory reclamation
                                sleep 10

                            done
                        done
                    done
                done 
            done
        done
    done
done
