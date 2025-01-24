#!/bin/bash

#SBATCH --time=2-00:00:00 --mem=50G

# Initialize Conda
source ~/hpc-share/anaconda3/etc/profile.d/conda.sh

# Activate the desired environment
conda activate l2xgnn

for SEED in 0
do
    for FOLD in 3
    do
        for ALGORITHM in RBRICS
        do
        
            for LAYERTYPE in GINConv GCNConv GATConv
            do
                for MODELTYPE in SingleChannel
                do 
                    for DATASETNAME in Lipophilicity
                    do
                        FOLDER="EXPT-12R-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-$MODELTYPE-$ALGORITHM"
                        mkdir $FOLDER
                        python 1_run_EXPT_SingleChannel.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type Regression --model_type $MODELTYPE --size_reg 0.0 --date_tag 1225 --epochs 500 --output_dir $FOLDER > $FOLDER/EXPT-out.out 2>&1
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
