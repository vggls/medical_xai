from collections import OrderedDict
from torchvision import models
from torch import nn

class ResNeXt():
    
    '''
    ATTRIBUTES: see README_attributes.md
    '''
    
    def __init__(self, type_, no_of_classes, trainable_layers=None, custom_classifier=None):
        
        self.type_ = type_
        self.no_of_classes = no_of_classes
        self.trainable_layers = trainable_layers
        self.custom_classifier = custom_classifier
        
        if self.type_ == '50_32x4d':
            self.model = models.resnext50_32x4d(pretrained=True)
        elif self.type_ == '101_32x8d':
            self.model = models.resnext101_32x8d(pretrained=True)
        
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
        all_layers = [self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
                    self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        if self.trainable_layers==None:
            self.freeze = all_layers
        else: 
            assert all(x in range(len(all_layers)) for x in self.trainable_layers)
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