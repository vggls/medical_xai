from torch.utils.data import DataLoader
from training_loop import Train
    
def fit(train_dataset, validation_dataset, batch_size,
        model, loss_fct, optimizer, scheduler,
        epochs, patience,
        no_of_classes, labels_of_normal_classes,
        train_sampler=None):
    
    '''
    ARGUMENTS: 
        
    --> 'train_dataset', 'validation_dataset', 'batch_size, 'train_sampler' arguments for DataLoader instance
         More specifically:
             - train_dataset and validation_dataset: variables as extracted by ImageFolder method or CustomDataset class instance 
             - batch_size: integer
             - train_sampler: Default is none. Otherwise it should be WeightedRandomSampler instance. 
                              Can be created via get_sampler method from imbalanced.py as well.
                              It is passed in the sampler argument of DataLoaders.
             
    --> 'model', 'loss_fct', 'optimizer_fct', 'learning rate', 'epochs', 'patience' arguments related to model's training 
         More specifically:
            - model: custom written model instance as per 'models' folder
            - loss_fct: The loss function. A torch.nn instance
            - optimizer: The algorithm to update weights. A torch.optim instance
            - scheduler: Use to change learning rate per epochs. A torch.optim instance
            - epochs: integer
            - patience: integer or 'None' to enable Early Stopping regularization
            
     --> 'no_of_classes': integer
         'labels_of_normal_classes': Should be either 'None' or a list of integers 
                                      ex. 0,1,2,3 for a 4-class problem
                                      The ImageFolder method and or CustomDataset class assigns integers automatically to the classes
                                      In order to verify the the class name-label assignment created by the Dataloader you may type
                                      "print(train_dataset.class_to_idx)"
            
    OUTPUTS:
    
    --> training_dict and validation_dict: dictionary to monitor loss, class and overall metrics
        
    '''
    
    # Dataloaders for training and validation datasetss
    if train_sampler==None:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=2, sampler=train_sampler)
    val_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # training_loop.py instance
    instance = Train(model.model, loss_fct, optimizer, scheduler,
                  train_loader, val_loader, 
                  epochs, patience,
                  no_of_classes, labels_of_normal_classes)

    training_dict, validation_dict = instance.training()

    return training_dict, validation_dict