#!/bin/bash

#SBATCH --time=2-00:00:00 --mem=50G

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
        
            for LAYERTYPE in GCN GAT GIN PNA SAGE
            do
                for MP_LAYERS in 2 3
                do
                    for HIDDEN_DIM in 8 16 32
                    do
                    
                        for DATASETNAME in tox21
                        do
                        
                            VanillaFOLDER="Vanilla_ML/EXPT-14ML-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-${LAYERTYPE}-MP${MP_LAYERS}-DIM${HIDDEN_DIM}-Vanilla-None"
                            python 1_run_EXPT_MultiChannel.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type MultiTask --model_type Vanilla --num_mp_layers $MP_LAYERS --hidden $HIDDEN_DIM --date_tag RBRICS0.005 --epochs 500 --output_dir $VanillaFOLDER
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
