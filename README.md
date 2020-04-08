# Feature_Extraction
This code performs feature extraction in Pytorch to allow to load pretrained models, remove final layers from the pretrained network and train them to fine tune for new dataset.
This code uses pretrained Chexpert model and retrains the final layers of the model on OpenI chest X ray dataset.
Also performs data augmentation, where the number of augmentation steps is dependent on the distribution of teh class(number of patients in that class) since the data is unbalanced. 
<p>
  <b> Arguments that need to be changed </b>
  <ul>
    <li> num_classes_orig: The number of output units/ classes in the original trained network.</li>
    <li> num_classes_final: The number of units in the network you want to train/fine-tune </li>
    <li> restore: The original pre-trained model. </li>
    <li> output_dir: Directory to store the training results and models. </li>
    </ul>
    </p>
    <p>
  <b> Requirements</b>
  <ul>
    <li> python 3.5+</li>
    <li>pytorch 1.0+</li>
    <li>torchvision</li>
    <li>numpy</li>
    <li>pandas</li>
    <li>sklearn</li>
    <li>matplotlib</li>
    <li>tensorboardX</li>
    <li>tqdm</li>
   </ul>
   </p>
   <p>
  Code for training and evaluation taken from https://github.com/kamenbliznashki/chexpert
  </p>

