'''
Official resnet implementation:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

'''

from collections import OrderedDict
from torchvision import models
from torch import nn
from torch.nn import ReLU, BatchNorm2d, Conv2d

class ResNet():
    
    '''
    ATTRIBUTES: see README_attributes.md
    
    Custom written class for ResNet34 and ResNet50 acrhitectures.
    
    The models are imported from torchvision and the following adjustments are made:
        1. We remove the last BatchNorm2d and ReLU layers such that the CNN part ends in a Conv2d layer.
        2. We replace a GAP layer with a Flatten layer. 
    '''
    
    def __init__(self, 
                 type_, 
                 no_of_classes, 
                 trainable_layers=None):
        
        self.type_ = type_
        self.no_of_classes = no_of_classes
        self.trainable_layers = trainable_layers
        
        if self.type_ == '34':
            resnet = models.resnet34(pretrained=True)
            flatten_nodes = 512 * 7 * 7
            layer4 = resnet.layer4
            layer4[2] = nn.Sequential(OrderedDict([
                ('conv1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ('bn1', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                ('relu', ReLU(inplace=True)),
                ('conv2', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
            ]))
            
        elif self.type_ == '50':
            resnet = models.resnet50(pretrained=True)
            flatten_nodes = 2048 * 7 * 7
            layer4 = resnet.layer4
            layer4[2] = nn.Sequential(OrderedDict([
                ('conv1', Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                ('bn1', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                ('conv2', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ('bn2', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                ('conv3', Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False))
            ]))
        
        resnet.avgpool = nn.Flatten()
        flatten = resnet.avgpool
        
        fc = nn.Linear(in_features=flatten_nodes, out_features=self.no_of_classes)
        
        self.model = nn.Sequential(OrderedDict([
            ("conv1", resnet.conv1),
            ("bn1", resnet.bn1),
            ("relu", resnet.relu),
            ("maxpool", resnet.maxpool),
            ("layer1", resnet.layer1),
            ("layer2", resnet.layer2),
            ("layer3", resnet.layer3),
            ("layer4", layer4),
            ("flatten", flatten),
            ("fc", fc)]
        ))
        
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