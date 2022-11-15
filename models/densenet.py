'''
Remark (**)

The following layer-index list is used to determine the 'trainable_feature_layers' attribute of the DenseNet class.
The attribute is either 'None' (default) or a 'list' of the below integers, 
so that the user can choose any layers of preference regardless of their position in the architecture.

The motivation for the definition of this attribute is that, in some cases,
we have observed that training the transition blocks and not the dense ones 
yields better results than training some of the last consequtive layers.

DenseNet architectures have the following 12(=len(model.features)) features layers:
    LAYER                     INDEX
model.features.conv0       :    0
model.features.norm0       :    1
model.features.relu0       :    2
model.features.pool0       :    3
model.features.denseblock1 :    4
model.features.transition1 :    5
model.features.denseblock2 :    6
model.features.transition2 :    7
model.features.denseblock3 :    8
model.features.transition3 :    9
model.features.denseblock4 :    10
model.features.norm5       :    11
'''

from collections import OrderedDict
from torchvision import models
from torch import nn

class DenseNet():
    
    '''
    ATTRIBUTES
    type_: '121' or '201' to load DenseNet121 or DenseNet201 respectively
    no_of_classes: integer
    trainable_feature_layers: as per above Remark (**)
    custom_classifier: Default 'None' value means that we consider the standard densenet classifier 
                       as imported by torchvision along with an additional Softmax layer.
                       Otherwise a custom classifier could be put on top of the model. 
    '''
    
    def __init__(self, type_, no_of_classes, trainable_feature_layers=None, custom_classifier=None):
        
        self.type_ = type_
        self.no_of_classes = no_of_classes
        self.layers = trainable_feature_layers
        self.custom_classifier = custom_classifier
        
        if self.type_ == '121':
            self.model = models.densenet121(pretrained=True)
        elif self.type_ == '201':
            self.model = models.densenet201(pretrained=True)
        
        # CLASSIFIER
        num_filters = self.model.classifier.in_features
        if self.custom_classifier==None:
            default_classifier = nn.Sequential(OrderedDict([
                ('0', nn.Linear(num_filters, self.no_of_classes)),
                ('1', nn.Softmax(dim=1))
            ]))
            self.model.classifier = default_classifier
        else:
            assert self.custom_classifier[0].in_features == num_filters
            assert self.custom_classifier[-2].out_features == self.no_of_classes
            assert type(custom_classifier[-1]) == nn.modules.activation.Softmax
            self.model.classifier = self.custom_classifier
        
        # LAYERS TO FREEZE DURING TRAINING
        if self.layers==None:
            self.freeze = self.model.features
        else: 
            self.freeze = [self.model.features[j] for j in range(12) if j not in self.layers]
            
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = False
        
        
    def trainable_params(self):
        # below is equivalent to summary from torchsummary
        print('No. of trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def unfreeze(self):
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = True