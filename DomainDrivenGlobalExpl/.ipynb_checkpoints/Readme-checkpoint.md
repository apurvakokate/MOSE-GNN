1. Class Count difference Ratio + Motif Params - Initial prior belief entropy (Incorrect implementation where param sigmoid was used and prior's raw value) -> 0.79 Accuracy 0.49 Val Loss. Motif params get dispersed seemingly move to zero

2. Class Count difference Ratio + Motif Params - Initial prior belief entropy  -> 0.8839 Accuracy 0.3178 (LR = 0.0001) Val Loss, mask loss keeps reducing .3186 to .3093, Motif params don't disperse as much

    Positives - Good Accuracy and loss
                separability in class specific motifs
                
    Negatives - Heavily relies on separability in the dataset
                Any model changes to motif parameters will be overridden by the loss calculation using priors
                We're always pulling weights closer to priors
                Can't handles unknown motifs
                
3. For the above approach, when we use randomly initialized weights, the points move very slowly. After 30 epochs, the the distribution is still random -> .73 acccuracy 0.53 Val Loss

4. For the above approach when we user random with higher learning rate, the motif weights start to converge but accuracy suffers significantly as the distribution is changing constantly -> 0.671 0.5738 (LR 0.0006 + Size loss)

5. For batchwise approach, no spread of the distribution (explainer params), the model sees the best masking scheme, heavily biased by the data distribution (count differences), low reg loss, NO2 does not go to the top -> 0.88 0.29  

For results before JUly 11th model had bugs (x used instead of x0) so the performance is different

6. Using a combination of class discriminative counts per motif and absolute class-wise counts as domain driven priors. SHould be robust to datasets. Based on size loss and entropy loss move points towards 0,0, 1,0, and 0,1 1,1. But probabibility of 1,1 is reduced by nature being on y=-x line. -> epoch 47  Val Loss: 0.2870 Val Acc: 0.88 for Mutagenicity. Improvement: Anti Sigmoid, increase explainer learning rate

7. When reverse sigmoid is applied the lower limit is set at -2,-2 . Val loss stagnates at Val Loss: 0.4321, Train Acc: 0.7691054261039596, Val Acc: 0.7770534550195567 Loss : 1.0433655920781588 , Mask Loss :  0.6148631848787006 after 118 epochs

8. If parameters are initiazed based on independent score calculation they follow y=-x line but are moved from here depending on cllass wise counts. 

9. hERG is initially overfitting although expl have good separation. Accuracy 74%

10. Setting unknown Test motifs to zero with 50% sampling:Fidelity + is lower which is bad since complement scores arent meaningful. Training and val performance remains the same

11. If the importance is viewed from single class for single channel, it doesn't matter which clas but the model will always be 42~45% accurate. We need to show both sides of prior.

12. Noise affect fidelity


# Update Aug 24
I have a week to work on experiments. SIgmoid activated weights dont help themodel learn classification. ANd the current config contain incorrect hyperparameters.


#Todo

1. Update train_model_params code to only use "model" variable -> EXP6
2. Make model and motif epochs dynamic, currently set to 20 -> EXPT6
3. Maybe refresh criterion every call as well -> NOT REQUIRED
4. does setting to model.train, reset ayour requires_grad to trueS -> No
5. Should the learning rate be reduced after each converged -> EXP6

6. Removed epoch check for early stopping - EXP6

7. check in model code if self.ones is getting set - Done
8. lookup_dict = self.lookup if all(smile in self.lookup for smile in smiles) else self.test_lookup, what if smile missing from lookup during train
9. STL
10. plot motof importance by class
11. Unit test original_pred_y logic


Todo :
RUN SME

Get class info in motif importance

Final model not saved for expt 4, roc calculated for it

If early stop in expt 5 and 6, maybe reduce max alternations. Or reduce patience

Expt5 and 6 are not loading the best model before evaluation