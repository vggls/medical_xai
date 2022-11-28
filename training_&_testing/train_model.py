from torch.utils.data import DataLoader
from training_loop import Train
    
def fit(train_dataset, validation_dataset, batch_size,
        model, loss_fct, optimizer,
        epochs, patience,
        no_of_classes, labels_of_normal_classes,
        train_sampler=None,
        scheduler=None):
    
    '''
    ARGUMENTS: 
        
    --> 'train_dataset', 'validation_dataset', 'batch_size, 'train_sampler' arguments for DataLoader instance
         More specifically:
             - train_dataset and validation_dataset: variables as extracted by ImageFolder method or CustomDataset class instance 
             - batch_size: integer
             - train_sampler: Default is none. Otherwise it should be WeightedRandomSampler instance. 
                              Can be created via get_sampler method from imbalanced.py as well.
                              It is passed in the sampler argument of DataLoaders.
             
    --> 'model', 'loss_fct', 'optimizer_fct', 'learning rate' arguments related to model's training 
         More specifically:
            - model: custom written model instance as per 'models' folder
            - loss_fct: The loss function. A torch.nn instance
            - optimizer: The algorithm to update weights. A torch.optim instance
            - scheduler: Use to change learning rate per epochs. A torch.optim instance
            
     --> 'no_of_classes': integer
         'labels_of_normal_classes': Should be either 'None' or a list of integers 
                                      ex. 0,1,2,3 for a 4-class problem
                                      The ImageFolder method and or CustomDataset class assigns integers automatically to the classes
                                      In order to verify the the class name-label assignment created by the Dataloader you may type
                                      "print(train_dataset.class_to_idx)"
            
    OUTPUTS:
    
    --> training_dict and validation_dict: you may find a detailed description in the training_loop.py file 
        
    '''
    
    # Dataloaders for training and validation datasetss
    if train_sampler==None:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=2, sampler=train_sampler)
    val_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # training_loop.py instance
    instance = Train(model.model, loss_fct, optimizer, 
                  train_loader, val_loader, 
                  epochs, patience,
                  no_of_classes, labels_of_normal_classes)
    instance.scheduler = scheduler #default scheduler attribute Train class value is 'None' 

    training_dict, validation_dict = instance.training()

    return training_dict, validation_dict