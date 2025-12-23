#!/bin/bash

#SBATCH --time=2-00:00:00 --mem=128G
#SBATCH --gres=gpu:1 

# Initialize Conda
source ~/hpc-share/anaconda3/etc/profile.d/conda.sh

# Activate the desired environment
conda activate l2xgnn

DICT="/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/Tmp_DICTIONARY"

for SEED in 0 
do
    for FOLD in 0 1 2 3 4
    do
        for ALGORITHM in RBRICS 
        do
            
            for LAYERTYPE in GAT GCN SAGE GIN
            do
                for HIDDEN_DIM in 32
                do
                    for EXPL_LR in 0.01
                    do
                        for GNN_LR in 0.001
                        do
                            for SIZE_REG in 0.0 0.00005 0.0005
                            do
                                for ENT_REG in 0.0 0.1 0.2
                                do
                                    for MP_LAYERS in 3
                                    do

                                        for DATASETNAME in hERG
                                        do

                                            FOLDER="/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/All0.5_learn_unk_hyperparam/RBRICS_MINORITY_MP${MP_LAYERS}_DIM${HIDDEN_DIM}_GNNLR${GNN_LR}/MOSE_BC/EXPT-14BC-$DATASETNAME-SEED-$SEED-FOLD-$FOLD-${LAYERTYPE}-EXPLLR-${EXPL_LR}-ENT-${ENT_REG}-SIZE-${SIZE_REG}-SingleChannel-${ALGORITHM}"
                                            

                                            python 1_run_EXPT_SingleChannel.py --dataset_name $DATASETNAME --seed $SEED --fold $FOLD --learn_unknown --algorithm $ALGORITHM --layer_type $LAYERTYPE --task_type BinaryClass --model_type SingleChannel --num_mp_layers ${MP_LAYERS} --hidden ${HIDDEN_DIM} --patience 30 --ent_reg ${ENT_REG} --size_reg ${SIZE_REG} --base_importance 0 --unk_importance 0 --expl_lr $EXPL_LR --lr $GNN_LR --date_tag RBRICS0.5_minority_True --epochs 500 --use_zero_weight --output_dir $FOLDER --path $DICT

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
        done
    done
done

