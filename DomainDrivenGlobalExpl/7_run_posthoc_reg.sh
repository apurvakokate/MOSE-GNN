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
                for MODELTYPE in SingleChannel
                do 
                    for DATASETNAME in Lipophilicity
                    do
                        FOLDER="0205_ALL/0205Lipo/EXPT-12R-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-$MODELTYPE-$ALGORITHM"
                        VanillaFOLDER="0205_ALL/Vanilla_Reg/EXPT-12R-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-Vanilla-None"
                        python 4_run_posthoc.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type Regression --model_type $MODELTYPE --size_reg 0.0 --date_tag 0205 --epochs 500 --output_dir $FOLDER --vanilla_dir $VanillaFOLDER
                        # Kill any lingering Python processes
                        pkill -f 4_run_posthoc.py
                        # Pause to allow memory reclamation
                        sleep 10
                    done
                    for DATASETNAME in esol
                    do
                        FOLDER="0205_ALL/0205esol/EXPT-12R-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-$MODELTYPE-$ALGORITHM"
                        VanillaFOLDER="0205_ALL/Vanilla_Reg/EXPT-12R-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-Vanilla-None"
                        python 4_run_posthoc.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type Regression --model_type $MODELTYPE --size_reg 0.0 --date_tag 0205 --epochs 500 --output_dir $FOLDER --vanilla_dir $VanillaFOLDER
                        # Kill any lingering Python processes
                        pkill -f 4_run_posthoc.py
                        # Pause to allow memory reclamation
                        sleep 10
                    done
                done

            done
        done
    done
done
