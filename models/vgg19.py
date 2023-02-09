import collections
#import torch
from torchvision import models
import torch.nn as nn

class VGG19():
    
    '''
    ATTRIBUTES: see README_attributes.md
    
    Custom written class for VGG19 architecture.
    
    The model is imported from torchvision and the following adjustments are made:
        1. Remove last ReLU and MaxPooling layers such that the CNN part ends in a Conv2d layer
        2. Replace the GAP layer by a Flatten layer
        3. Replace the original classifier by a simple nn.Linear layer that yield the raw class scores
    '''
    
    def __init__(self, 
                 num_classes, 
                 trainable_feature_layers=None):
        
        self.num_classes = num_classes
        self.trainable_feature_layers = trainable_feature_layers
        
        vgg19 = models.vgg19(pretrained=True)
        
        features = vgg19.features[:-2]
        
        vgg19.avgpool = nn.Flatten()
        flatten = vgg19.avgpool
        
        classifier = nn.Linear(512 * 14 * 14, self.num_classes)
        
        self.model = nn.Sequential(collections.OrderedDict([
            ("features", features),
            ("flatten", flatten),
            ("classifier", classifier)]
        ))

        # LAYERS TO FREEZE DURING TRAINING
        if self.trainable_feature_layers==None:
            self.freeze = self.model.features
        else: 
            assert all(x in range(len(self.model.features)) for x in self.trainable_feature_layers)
            self.freeze = [self.model.features[j] for j in range(len(self.model.features)) \
                           if j not in self.trainable_feature_layers]
            
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = False

        
    def trainable_params(self):
        print('No. of trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        
    
    def unfreeze(self):
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = True
