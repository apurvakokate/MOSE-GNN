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
                for ENT in 0.0 0.1
                do 
                    for DATASETNAME in hERG
                    do
                        FOLDER="ent_reg/EXPT-ENTBC-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-$ENT-$ALGORITHM"
                        python 4_run_vanilla_compare.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type BinaryClass --model_type SingleChannel --size_reg 0.0 --ent_reg $ENT --date_tag 0205 --epochs 500 --output_dir $FOLDER --vanilla_dir None
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
