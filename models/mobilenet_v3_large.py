from collections import OrderedDict
from torchvision import models
from torch import nn

class MobileNet_V3_Large():
    
    '''
    ATTRIBUTES
    no_of_classes: integer
    trainable_feature_layers: The mobilenet_v3_large consists of 17 feature layers, indexed 0 to 16.
                        This attribute is either 'None' (by default) if we want to freeze all the layers during training
                        OR a list of the feature layer indices that we want to train.
                        Note that the classifier part is always trained.
    custom_classifier: Default 'None' value means that we consider the standard mobilenet classifier 
                       as imported by torchvision along with an additional Softmax layer.
                       Otherwise a custom classifier could be put on top of the model. 
    '''
    
    def __init__(self, no_of_classes, trainable_feature_layers=None, custom_classifier=None):
        
        self.no_of_classes = no_of_classes
        self.trainable_feature_layers = trainable_feature_layers
        self.custom_classifier = custom_classifier
        
        self.model = models.mobilenet_v3_large(pretrained=True)
        
        # CLASSIFIER
        num_filters = self.model.classifier[0].in_features
        if self.custom_classifier==None:
            default_classifier = nn.Sequential(OrderedDict([
                ('0', nn.Linear(in_features=num_filters, out_features=1280, bias=True)),
                ('1', nn.Hardswish()),
                ('2', nn.Dropout(p=0.2, inplace=True)),
                ('3', nn.Linear(in_features=1280, out_features=self.no_of_classes, bias=True)),
                ('4', nn.Softmax(dim=1))
            ]))
            self.model.classifier = default_classifier
        else:
            assert self.custom_classifier[0].in_features == num_filters
            assert self.custom_classifier[-2].out_features == self.no_of_classes
            assert type(custom_classifier[-1]) == nn.modules.activation.Softmax
            self.model.classifier = self.custom_classifier
        
        # LAYERS TO FREEZE DURING TRAINING
        if self.trainable_feature_layers==None:
            self.freeze = self.model.features
        else: 
            len_ = len(self.model.features) #17
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