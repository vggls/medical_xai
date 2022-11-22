This repo is organized as follows:

- **models** folder <br/>
  
   - **densenet.py**, **vgg.py**, **efficientnet.py**, **googlenet.py** :  custom classes which instantiate pretrained models with adjustable frozen layers and custom classifiers

- **training & testing** folder <br/>
    
    - **imbalanced.py** : Creates class weights and a WeighterRandomSampler instance to address class imbalance in the dataset
    - **training_loop.py** : Training loop class implementation for a NN model. Per epoch we compute the loss and class metrics of the training and validation phase.
          In addition the following regularization techniques are included : <br/>
          a) Early Stopping regularization tenchnique. Note that since the code is written for medical tasks purposes, the method monitors improvements on the validation loss and the average validation recall scores of the disease related classes. <br/>
          b) Scheduled control of the optimizer learning rate. Application is optional via the 'scheduler' attribute.
          
    - **train_model.py** : Includes the 'fit' method which loads data via DataLoaders and trains a model according to training_loop.py
    - **testing_report.py** : Implements classification report and ROC and PR curves for given model and dataloader object

- **heatmap.py** : Includes function that accepts pixel-level attributions obtained by XAI algorithm (ex. HiResCAM) and calculates 
    
    - A region-level heatmap, named 'heatmap', which emerges by applying AvgPooling transformation on the pixel attributions.
    
    - A list of the 'heatmap' regions in descending order of importance. The list is named 'regions'.

    We note that 'heatmap' and 'regions' will serve as the main tools for the calculation of the xai evaluation metric called AOPC, in *morf.py*

- **overlay.py** : In this file we generate a super-imposed version of the original image by adding a weighted version of the XAI algorithm heatmap.

- **morf.py** : For a given tensor and model, the 'MoRF' class implements the MoRF tenchnnique for heatmap evaluation and calculates the AOPC score. The file also includes a method that extends the calculation on a dataset level, when the data are called via Dataloaders.

- **haas.py** : Calculation of the HAAS score for evaluation of XAI algorithms

Remark: At the beginning of each .py file, in the comments section, we have included the sources used (theory, code etc) along with remarks and the main ideas, where necessary.
