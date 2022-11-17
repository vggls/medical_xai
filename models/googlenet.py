'''
Remark (**)

The following layer-index list is used to determine the 'trainable_layers' attribute of GoogLeNet class.
The attribute is either 'None' (default) or a 'list' of the below integers, 
so that the user can choose any layers of preference regardless of their position in the architecture.

  LAYER         INDEX
conv1        :    0 
maxpool1     :    1
conv2        :    2
conv3        :    3
maxpool2     :    4
inception3a  :    5
inception3b  :    6
maxpool3     :    7
inception4a  :    8
inception4b  :    9
inception4c  :   10 
inception4d  :   11
inception4e  :   12
maxpool4     :   13
inception5a  :   14
inception5b  :   15

'''

from collections import OrderedDict
from torchvision import models
from torch import nn

class GoogLeNet():
    
    '''
    ATTRIBUTES
    no_of_classes: integer
    trainable_layers: As per above Remark (**). Note that 'fc' layer is always trainable.
    custom_classifier: Default 'None' value means that we consider the standard googlenet classifier 
                       as imported by torchvision along with an additional Softmax layer.
                       Otherwise a custom classifier could be put on top of the model. 
    '''
    
    def __init__(self, no_of_classes, trainable_layers=None, custom_classifier=None):
        
        self.no_of_classes = no_of_classes
        self.layers = trainable_layers
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
            assert self.custom_classifier[0].in_features == num_filters
            assert self.custom_classifier[-2].out_features == self.no_of_classes
            assert type(custom_classifier[-1]) == nn.modules.activation.Softmax
            self.model.fc = self.custom_classifier
        
        # LAYERS TO FREEZE DURING TRAINING
        all_layers = [self.model.conv1, self.model.maxpool1, self.model.conv2, self.model.conv3,
                    self.model.maxpool2, self.model.inception3a, self.model.inception3b, self.model.maxpool3,
                    self.model.inception4a, self.model.inception4b, self.model.inception4c, 
                    self.model.inception4d, self.model.inception4e, self.model.maxpool4,
                    self.model.inception5a, self.model.inception5b]
        if self.layers==None:
            self.freeze = all_layers
        else: 
            self.freeze = [all_layers[j] for j in range(len(all_layers)) if j not in self.layers]
            
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = False
        
        
    def trainable_params(self):
        print('No. of trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def unfreeze(self):
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = True