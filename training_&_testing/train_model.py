import torch
from torch.utils.data import DataLoader
from training_loop import Train
    
def fit(train_dataset, validation_dataset, batch_size, train_sampler,
        model, loss_fct, optimizer_fct, learning_rate,
        no_of_classes, label_of_normal_class):
    
    '''
    ARGUMENTS: 
        
    --> 'train_dataset', 'validation_dataset', 'batch_size, 'train_sampler' arguments for DataLoader instance
         More specifically:
             - train_dataset and validation_dataset: variables as extracted by ImageFolder method or CustomDataset class instance 
             - batch_size
             - train_sampler: a WeightedRandomSampler instance. Can be created via sampler.py as well
             
    --> 'model', 'loss_fct', 'optimizer_fct', 'learning rate' arguments related to model's training 
         More specifically:
            - model: custom written model instance as per 'models' folder
            - loss_fct: The loss function. A torch.nn instance
            - optimizer_fct: one of 'Adam' or 'SGD'
            - learning_rate: the optimizer's learning rate argument
            
     --> 'no_of_classes': integer
         'label_of_normal_class': Labels should be in integers (not strings) starting from 0 
                                  ex. 0,1,2,3 for a 4-class problem
                                  The ImageFolder method and or CustomDataset class assigns integers automatically to the classes
                                  In order to verify the the class name-label assignment created by the Dataloader you may type
                                  "print(train_dataset.class_to_idx)"
            
    OUTPUTS:
    
    --> training_dict and validation_dict: you may find a detailed description in the training_loop.py file 
        
    '''
    
    # Dataloaders
    if train_sampler==None:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=2, sampler=train_sampler)
    val_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
    # Optimizer
    if optimizer_fct == 'Adam':
        optimizer = torch.optim.Adam(model.model.parameters(), lr = learning_rate)
    elif optimizer_fct == 'SGD':
        optimizer = torch.optim.SGD(model.model.parameters(), lr = learning_rate)
    
    # training_loop.py instance
    instance = Train(model.model, loss_fct, optimizer, 
                  train_loader, val_loader, 
                  no_of_classes, label_of_normal_class)

    training_dict, validation_dict = instance.training()

    return training_dict, validation_dict


'''
EXAMPLE CODE

ex1: how to call the 'fit' function

# model instance, as per efficientnet.py custom written file in 'models' folder
efficientnet_b4 = EfficientNet(type_ = 'b4',  no_of_classes = 4) 

training_dict, validation_dict = fit(train_dataset,
                                    validation_dataset,
                                    batch_size = 128,
                                    train_sampler=None,
                                    model = efficientnet_b4,
                                    loss_fct = torch.nn.CrossEntropyLoss(),
                                    optimizer_fct = 'Adam',
                                    learning_rate = 0.001,
                                    no_of_classes = 4,
                                    label_of_normal_class = 2)
--------------------------------------------------------------------------

ex2: how to save the dictionaries in pickle files:

import pickle    
path = ...
with open(path + 'training_dict.pickle', 'wb') as f: 
  pickle.dump(training_dict, f)
with open(path + 'validation_dict.pickle', 'wb') as f: 
  pickle.dump(validation_dict, f)
'''