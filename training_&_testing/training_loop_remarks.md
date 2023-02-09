Remarks on training_loop.py

  1. Regularizers
      - Early Stopping method: See remark 2 below for the condition it uses. The method is enabled by the 'patience' attribute, which is either integer or 'None'.
      - 'scheduler' attribute: Either 'None' or a https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html instance.

  2. The training loop ('training' method) focuses on training a *medical task model*.
      This means that we are mainly interested in the recall scores of the unhealthy classes.
      
      For this purpose, the callback function is an Early Stopping technique focusing on 
      the improvement of the validation loss and avg recall of the 'unhealthy' classes.
      
      By 'unhealthy' classes we refer to the non-normal classes of a dataset.
      
      In ordet to determine the 'unhealthy' classes the 'labels_of_normal_classes' attribute should be used.
      
      For instance, consider a dataset with classes labelled by 0,1,2 where 0 and 2 are cancerous cells and 1 is a healthy cell.
      This attribute allows to stop training only when the cancerous classes are well classified witout being affected by the normal
      class results. Thus, in this scenario one may set labels_of_normal_classes = [1]

  3. The resulting best model is saved in a .pt file

  4. The 'training' method returns two dictionaries that contain the loss and metrics history 
      for the training and validation phases respectively.
      Each dictionary has the following self-explanatory keys: 
          - 'loss', 'accuracy', 'avg_recall', 'avg_precision', 'avg_f1'; 
              and the values are lists of the respective epoch values
          - 'recall_per_class', 'precision_per_class', 'f1_per_class';
              and the values are lists which consist of sublists equal to the number of classes.
              Each sublist describes the class metric history per epoch
