'''
Remark (**)

The following layer-index list is used to determine the 'trainable_feature_layers' attribute of the DenseNet class.
The attribute is either 'None' (default) or a 'list' of the below integers, 
so that the user can choose any layers of preference regardless of their position in the architecture.

The motivation for the definition of this attribute is that, in some cases,
we have observed that training the transition blocks and not the dense ones 
yields better results than training some of the last consequtive layers.

DenseNet architectures have the following 12 features layers:
    LAYER               INDEX
features.conv0       :    0
features.norm0       :    1
features.relu0       :    2
features.pool0       :    3
features.denseblock1 :    4
features.transition1 :    5
features.denseblock2 :    6
features.transition2 :    7
features.denseblock3 :    8
features.transition3 :    9
features.denseblock4 :    10
features.norm5       :    11
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
        self.trainable_feature_layers = trainable_feature_layers
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
            assert self.custom_classifier[-2].out_features == self.no_of_classes
            assert type(custom_classifier[-1]) == nn.modules.activation.Softmax
            self.model.classifier = self.custom_classifier
        
        # LAYERS TO FREEZE DURING TRAINING
        if self.trainable_feature_layers==None:
            self.freeze = self.model.features
        else:
            len_ = len(self.model.features)
            assert all(x in range(len_) for x in self.trainable_feature_layers)
            self.freeze = [self.model.features[j] for j in range(12) if j not in self.trainable_feature_layers]
            
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = False
        
        
    def trainable_params(self):
        print('No. of trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def unfreeze(self):
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = True
