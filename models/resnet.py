from collections import OrderedDict
from torchvision import models
from torch import nn

class ResNet():
    
    '''
    ATTRIBUTES: see README_attributes.md
    '''
    
    def __init__(self, 
                 type_, 
                 no_of_classes, 
                 trainable_layers=None, 
                 flatten=True):
        
        self.type_ = type_
        self.no_of_classes = no_of_classes
        self.trainable_layers = trainable_layers
        self.flatten = flatten
        
        if self.type_ == '34':
            self.model = models.resnet34(pretrained=True)
            self.flatten_nodes = 512 * 7 * 7
        elif self.type_ == '50':
            self.model = models.resnet50(pretrained=True)
            self.flatten_nodes = 2048 * 7 * 7
        elif self.type_ == '101':
            self.model = models.resnet101(pretrained=True)
            self.flatten_nodes = 2048 * 7 * 7
        elif self.type_ == '152':
            self.model = models.resnet152(pretrained=True)
            self.flatten_nodes = 2048 * 7 * 7
        
        # REPLACE GAP WITH FLATTEN
        if self.flatten == True:
            self.model.avgpool = nn.Flatten() # haven't changed layer name though - to do
        
        # CLASSIFIER WITH SOFTMAX
        if self.flatten == False:
            num_filters = self.model.fc.in_features
        elif self.flatten == True:
            num_filters = self.flatten_nodes
        classifier = nn.Sequential(OrderedDict([
                ('0', nn.Linear(in_features=num_filters, 
                                out_features=self.no_of_classes)),
                ('1', nn.Softmax(dim=1))
            ]))
        self.model.fc = classifier

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