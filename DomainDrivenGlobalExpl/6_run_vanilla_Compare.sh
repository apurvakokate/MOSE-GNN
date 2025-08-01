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
        
            for LAYERTYPE in GATConv GINConv GCNConv
            do
                for ENT in 0.0
                do 
                    for DATASETNAME in Mutagenicity
                    do
                        FOLDER="0205_ENT/Mutagenicity/EXPT-13BC-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-$ENT-$ALGORITHM"
                        VanillaFOLDER="0205_ENT/Vanilla_BC/EXPT-12BC-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-Vanilla-None"
                        python 4_run_vanilla_compare.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type Regression --model_type SingleChannel --size_reg 0.0 --date_tag RBRICS0.005 --epochs 500 --output_dir $FOLDER --vanilla_dir $VanillaFOLDER
                        # Kill any lingering Python processes
                        pkill -f 4_run_vanilla_compare.py
                        # Pause to allow memory reclamation
                        sleep 10
                    done 
                done

            done
        done
    done
done
