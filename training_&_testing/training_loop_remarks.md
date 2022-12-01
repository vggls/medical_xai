Remarks on training_loop.py

  1. The training loop ('training' method) focuses on training a medical task model. 
      This means that we are mainly interested in the recall scores of the unhealthy classes.
      For this purpose, the callback function is an Early Stopping technique focusing on 
      the improvement of the validation loss and avg recall (see remark 7 as well) of the unhealthy classes (min_delta = 0) 

  2. Per training epoch we see/print the progress of the loss, accuracy and recall metrics.

  3. On 'scheduler' attribute: https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
      Note that when scheduler is included, per epoch the in-force learning rate is printed as well.

  4. The resulting best model is saved in a .pt file

  5. The 'training' method returns two dictionaries that contain the loss and metrics history 
      for the training and validation phases respectively.
      Each dictionary has the following self-explanatory keys: 
          - 'loss', 'accuracy', 'avg_recall', 'avg_precision', 'avg_f1'; 
              and the values are lists of the respective epoch values
          - 'recall_per_class', 'precision_per_class', 'f1_per_class';
              and the values are lists which consist of sublists equal to the number of classes.
              Each sublist describes the class metric history per epoch

  6. The attribute 'labels_of_normal_classes' can be used in case we want to regularize training wrt to specific classes.
      For instance, consider a dataset with classes labelled by 0,1,2 where 0 and 2 are cancerous cells and 1 is a healthy cell.
      This attribute allows to stop training only when the cancerous classes are well classified witout being affected by the normal
      class results. Thus, in this scenario one may set labels_of_normal_classes = [1]

  7. On the condition of the Early Stopping callback:
  
     - In the code we consider the 'average recall' of the 'un-normal' classes, where each class'es recall contributes the same to the final average. 
     
     - Alternatively, one may consider a 'weighted recall', where each classe'es recall is weighted by the class instances as well. In the end the
      resulting sum over all classes is divided the total number of instances.
      ex. for two classes, the weighted avg recall would be: weighted_avg_recall=(r1∗|c1|)+(r2∗|c2|)|c1|+|c2| ,
      where  r1  and  r2  are the recalls for class 1 and class 2, and  |c1|  and  |c2|  are the number of instances in class 1 and class 2.
      Note that above calculation is equal to the 'accuracy' metric score.
      
