"""
=> How to load train/val/test data via Dataloaders: 
    - Organize manually data into training_folder, validation_folder and test_folder
      such that each folder contains the data into separate subfolders with the class name.
    - Pass the folder path into the 'root' argument of the -ImageFolder- method and assign the result to a 'dataset' variable.
      Note that by doing this the labels are automatically assigned to integer numbers.
      As far as the 'transforms' argument is concerned note that the transforms of the validation and test data 
      should be the same and without any augmentation methods.
    - Finally, load data by inserting the 'dataset' to a -Dataloader- instance. 
      At this step, you may also pass a 'sampler' as explained below.
      
=> Sources for the WeightedRandomSampler' sampler:
    - https://www.youtube.com/watch?v=4JFVhJyTZ44&ab_channel=AladdinPersson   (code)
    - https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
    - https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample
    
=> How to use the 'oversample' argument of the get_loader function:
    - Apply to train data after train/val/test split
    - For training data use train_transforms and set "shuffle=False and oversample=True" or "shuffle=True and oversample=False"
    - For validation data use test_transforms and "shuffle=True and oversample=False"
    - For test data use test_transforms and set "shuffle=False and oversample=False"
    
"""

import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

def get_loader(dataset_dir, transforms, batch_size, shuffle=True, oversample=False):

    # WeightedRandomSampler samples randomly from a given dataset.
    assert not(shuffle==True and oversample==True)

    # ImageFolder
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transforms)
    #print(len(dataset))

    # sampler
    if oversample==False:
        sampler = None
    elif oversample==True:
        class_weights=[]; no_files = [] #;i=0
        for root, subdir, files in os.walk(dataset_dir):
            subdir.sort()
            if len(files)>0:
                no_files.append(len(files))
                class_weights.append(1/len(files))
                #print('Class {} has {} files and gets weight {}'. \
                #      format(list(dataset.class_to_idx.keys())[i], len(files), 1/len(files)))
                #i += 1
                
        sample_weights = [0]*len(dataset)
        for idx, (data, label) in enumerate(dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight
        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights), # len(sample_weights) = len(dataset)
                                        replacement=True)

    # Dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    return loader
