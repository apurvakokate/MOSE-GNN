EXPT4 - Training model to convergence with explainer false, Training explainer with gnn frozen
EXPT5 - Altering training with set convergengence iterations, reinit optimizer, and inner epochs without early stop coz 30 epochs never reached
EXPT 6 - Altering with dynamicly calculated iterations with global optimizer + plot per class, removed check for early stop, use explainer true in model init
EXPT 7 - Early stop based on fidelity and val loss, saving best model
Set motif params require grad to true and tturned it into leaf
EXPT 8 - 4 EXPT 4 except changes:
1. Start with using explainer but not train explainer
2. When training explainer also train GNN
EXPT 9 - EXPT 7 with joint training in explainer loop
EXPT11 - EXPT 7 with STL



-------------------------------------------
Steps:
0_CreateMotifVocab.ipynb -----> Creates files in dictionary/ 