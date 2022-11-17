from collections import OrderedDict
from torchvision import models
from torch import nn

class GoogLeNet():
    
    '''
    ATTRIBUTES: see README_attributes.md
    '''
    
    def __init__(self, no_of_classes, trainable_layers=None, custom_classifier=None):
        
        self.no_of_classes = no_of_classes
        self.trainable_layers = trainable_layers
        self.custom_classifier = custom_classifier
        
        self.model = models.googlenet(pretrained=True)
        
        # CLASSIFIER
        num_filters = self.model.fc.in_features
        if self.custom_classifier==None:
            default_classifier = nn.Sequential(OrderedDict([
                ('0', nn.Linear(num_filters, self.no_of_classes)),
                ('1', nn.Softmax(dim=1))
            ]))
            self.model.fc = default_classifier
        else:
            assert self.custom_classifier[-2].out_features == self.no_of_classes
            assert type(custom_classifier[-1]) == nn.modules.activation.Softmax
            self.model.fc = self.custom_classifier
        
        # LAYERS TO FREEZE DURING TRAINING
        all_layers = [self.model.conv1, self.model.maxpool1, self.model.conv2, self.model.conv3,
                    self.model.maxpool2, self.model.inception3a, self.model.inception3b, self.model.maxpool3,
                    self.model.inception4a, self.model.inception4b, self.model.inception4c, 
                    self.model.inception4d, self.model.inception4e, self.model.maxpool4,
                    self.model.inception5a, self.model.inception5b]
        if self.trainable_layers==None:
            self.freeze = all_layers
        else: 
            assert all(x in range(len(all_layers)) for x in self.trainable_feature_layers)
            self.freeze = [all_layers[j] for j in range(len(all_layers)) if j not in self.trainable_layers]
            
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = False
        
        
    def trainable_params(self):
        print('No. of trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def unfreeze(self):
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = True