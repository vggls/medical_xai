This repo is organized as follows:

- **xai_metrics** folder <br/>
    
    Contains implementation of AOPC, Max Sensitivity and HAAS metrics for evaluation of XAI attribution maps. The methods are customized to 'CAM' algorithms as imported by [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam) collection.

- **models** folder <br/>

    Custom written classes which instantiate pretrained models with adjustable no. of classes and trainable layers. Each model has the following structure : CNN ending in a Conv2d layer - Flatten - Raw class scores.

- **training & testing** folder <br/>
    
    - **imbalanced.py** : Creates class weights and a WeighterRandomSampler instance to address class imbalance in the dataset
    - **training_loop.py** : Training loop class implementation for a NN model. Per epoch we compute the loss and class metrics of the training and validation phase.
          In addition the following **regularization** optional techniques are included : <br/>
          a) Early Stopping regularization tenchnique. It is build with a focus on medical tasks and monitors improvements on the validation loss and the average validation recall scores of the disease related classes. <br/>
          b) Scheduled control of the optimizer learning rate.
          
    - **train_model.py** : Includes the 'fit' method which loads data via DataLoaders and trains a model according to training_loop.py
    - **testing_report.py** : Calculates classification report, balanced accuracy score and ROC and PR curves and scores for given model and dataloader object

- **utils** folder <br/>

    - **heatmap.py** : Method that converts a pixel-level attribution map (ex. as obtained by GradCAM) to a region-level attribution map.
    - **overlay.py** : Method that generates a super-imposed version of the original image by adding a weighted version of the XAI algorithm heatmap.
    - **plot_tensor.py**: Method that converts [-1,1]-valued tensor to [0,1]-valued tensor. To be used for plotting purposes.

- **example_xrays** folder <br/>

    Use of AOPC and Max Sensitivity scores in order to compare GradCAM and HiResCAM xai algorithms.

Remark: At the beginning of each .py file, in the comments section, there are sources (theory, code etc) along with remarks and the main ideas, where necessary.
