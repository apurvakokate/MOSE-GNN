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
        
            for LAYERTYPE in GCNConv GATConv GINConv
            do
                for MODELTYPE in MultiChannel
                do 
                    for DATASETNAME in tox21
                    do
                        FOLDER="0205_ALL/0205Tox/EXPT-12MT-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-$MODELTYPE-$ALGORITHM"
                        VanillaFOLDER="0205_ALL/Vanilla_Tox/EXPT-12MT-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-Vanilla-None"
                        mkdir $FOLDER
                        python 4_run_vanilla_compare_multi.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type MultiTask --model_type $MODELTYPE --size_reg 0.0 --date_tag 0205 --epochs 500 --output_dir $FOLDER --vanilla_dir $VanillaFOLDER
                        # Kill any lingering Python processes
                        pkill -f 4_run_vanilla_compare_multi.py
                        # Pause to allow memory reclamation
                        sleep 10
                    done 
                done

            done
        done
    done
done
