"""
This py file contains two approaches to deal with imbalanced datasets.

- 'class_weights' method assigns to each class a weight equal of the class inverse frequency.
    The calculated output weights should be passed to the loss function, via 'weight' argument, 
    in order to be used as penalties in the loss function during training.
    In that way the model tends to avoid false negatives over false positives.
    see also : https://datascience.stackexchange.com/questions/52627/why-class-weight-is-outperforming-oversampling
        
- 'get_sampler' method constructs a 'WeightedRandomSampler' instance which can be used to ovesample
    the minority class(es); by increasing the frequency that images from these classes are seen 
    by the model during training.
    The output of get_sampler method should be passed as the 'sampler' argument of the DataLoader.
    see also : https://www.youtube.com/watch?v=4JFVhJyTZ44&ab_channel=AladdinPersson   (code)
               https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
               https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample
"""

import os
from torch.utils.data import WeightedRandomSampler
import torch

def class_weights(dataset_dir, dataset):
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    weights=[]; no_files = [] ;i=0
    for root, subdir, files in os.walk(dataset_dir):
        subdir.sort()
        if len(files)>0:
            no_files.append(len(files))
            weights.append(1/len(files))
            print('{} class - instances {} - assigned weight {}'. \
                  format(list(dataset.class_to_idx.keys())[i], len(files), 1/len(files)))
            i += 1
    
    weights = torch.Tensor(weights)
    
    return weights

def get_sampler(dataset_dir, dataset):

    weights = class_weights(dataset_dir, dataset).tolist()
            
    sample_weights = [0]*len(dataset)
    for idx, (data, label) in enumerate(dataset):
        class_weight = weights[label]
        sample_weights[idx] = class_weight
        
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights), # len(sample_weights) = len(dataset)
                                    replacement=True)

    return sampler