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
        
            for LAYERTYPE in GCN GAT SAGE PNA
            do
                for ENT in 0.0 0.2
                do 
                    for DATASETNAME in Fluoride_Carbonyl
                    do
                        FOLDER="EXPT-12BC-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-$LAYERTYPE-$ENT-$ALGORITHM"
                        mkdir $FOLDER
                        python 1_run_EXPT_SingleChannel.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type BinaryClass --model_type SingleChannel --size_reg 0.0 --date_tag RBRICS0.005 --ent_reg $ENT --epochs 500 --output_dir $FOLDER > $FOLDER/EXPT-out.out 2>&1
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
