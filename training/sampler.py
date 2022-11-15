"""
=> The 'dataset' argument will be created either via 
    - the ImageFolder method or
    - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
   It depends on the data format
      
=> Sources for the WeightedRandomSampler' sampler:
    - https://www.youtube.com/watch?v=4JFVhJyTZ44&ab_channel=AladdinPersson   (code)
    - https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
    - https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample

=> To be passed as a DataLoader argument in case oversampling of training data is needed

"""

import os
from torch.utils.data import WeightedRandomSampler

def get_sampler(dataset_dir, dataset):

    class_weights=[]; no_files = [] ;i=0
    for root, subdir, files in os.walk(dataset_dir):
        subdir.sort()
        if len(files)>0:
            no_files.append(len(files))
            class_weights.append(1/len(files))
            print('{} class - instances {} - assigned weight {}'. \
                  format(list(dataset.class_to_idx.keys())[i], len(files), 1/len(files)))
            i += 1
            
    sample_weights = [0]*len(dataset)
    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
        
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights), # len(sample_weights) = len(dataset)
                                    replacement=True)

    return sampler