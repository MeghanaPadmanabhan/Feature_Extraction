# Feature_Extraction
This code performs feature extraction in Pytorch to allow to load pretrained models, remove final layers from the pretrained network and train them to fine tune for new dataset.
This code uses pretrained Chexpert model and retrains the final layers of the model on OpenI chest X ray dataset.
Also performs data augmentation, where the number of augmentation steps is dependent on the distribution of teh class(number of patients in that class) since the data is unbalanced. 

