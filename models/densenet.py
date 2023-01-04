from collections import OrderedDict
from torchvision import models
from torch import nn

class DenseNet():
    
    '''
    ATTRIBUTES: see README_attributes.md
    
    Custom written ResNet class.
    
    The models are imported from torchvision and the following adjustments are made:
        1. After the last Conv2d layer, we replace last BatchNorm2d by Flattened layer
        2. Add Softmax after the Linear layer

    '''
    
    def __init__(self, 
                 type_, 
                 no_of_classes, 
                 trainable_feature_layers=None):
        
        self.type_ = type_
        assert self.type_ in ['121', '169', '201']
        self.no_of_classes = no_of_classes
        self.trainable_feature_layers = trainable_feature_layers
        
        if self.type_ == '121':
            self.model = models.densenet121(pretrained=True)
            self.flatten_nodes = 1024 * 7 * 7
            
        elif self.type_ == '169':
            self.model = models.densenet169(pretrained=True)
            self.flatten_nodes = 1664 * 7 * 7
            
        elif self.type_ == '201':
            self.model = models.densenet201(pretrained=True)
            self.flatten_nodes = 1920 * 7 * 7

        # 1. REMOVE LAST BATCH NORM LAYER
        self.model.features = nn.Sequential(*[self.model.features[i] for i in range(len(self.model.features)-1)])

        # 2. SET CLASSIFIER WITH FLATTEN AND SOFTMAX
        classifier = nn.Sequential(OrderedDict([
                ('flatten', nn.Flatten()),
                ('linear', nn.Linear(in_features=self.flatten_nodes, out_features=self.no_of_classes)),
                ('softmax', nn.Softmax(dim=1))
            ]))
        self.model.classifier = classifier
        
        # PUT FEATURES AND CLASSIFIER TOGETHER
        self.model = nn.Sequential(OrderedDict([
                        ('features', self.model.features),
                        ('classifier', self.model.classifier)
                    ]))
        
        # LAYERS TO FREEZE DURING TRAINING
        if self.trainable_feature_layers==None:
            self.freeze = self.model.features
        else:
            len_ = len(self.model.features) #11
            assert all(x in range(len_) for x in self.trainable_feature_layers)
            self.freeze = [self.model.features[j] for j in range(len_) if j not in self.trainable_feature_layers]
           
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = False

        
    def trainable_params(self):
        print('No. of trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def unfreeze(self):
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = True