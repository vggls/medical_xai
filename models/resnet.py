from collections import OrderedDict
from torchvision import models
from torch import nn
from torch.nn import ReLU, BatchNorm2d, Conv2d

class ResNet():
    
    '''
    ATTRIBUTES: see README_attributes.md
    
    Custom written ResNet class.
    
    The models are imported from torchvision and the following adjustments are made:
        1. Remove last BatchNorm2d and ReLU layers such that the CNN part ends in a Conv2d layer
        2. Remove GAP layer. It is replaced with a flattened one coming from the last Conv2d layer
        3. Add Softmax after the Linear layer
            Rmk: Due to the flattened layer the in_features of the Linear layer should change as well. 
                One may use the summary method from torchvision to determine the number of nodes in the flattened layer.
    '''
    
    def __init__(self, 
                 type_, 
                 no_of_classes, 
                 trainable_layers=None):
        
        self.type_ = type_
        self.no_of_classes = no_of_classes
        self.trainable_layers = trainable_layers
        
        if self.type_ == '34':
            
            self.model = models.resnet34(pretrained=True)
            self.flatten_nodes = 512 * 7 * 7
            # 1. REMOVE LAST BATCH NORM LAYER that exists in the torchvision structure
            self.model.layer4[2] = nn.Sequential(OrderedDict([
                ('conv1', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ('bn1', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                ('relu', ReLU(inplace=True)),
                ('conv2', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
            ]))
            
        elif self.type_ == '50':
            
            self.model = models.resnet50(pretrained=True)
            self.flatten_nodes = 2048 * 7 * 7
            # 1. REMOVE LAST BATCH NORM and RELU LAYER that exist in the torchvision structure
            self.model.layer4[2] = nn.Sequential(OrderedDict([
                ('conv1', Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                ('bn1', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                ('conv2', Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ('bn2', BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                ('conv3', Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False))
            ]))

        
        # 2. REPLACE GAP WITH FLATTEN
        # haven't changed layer name though
        self.model.avgpool = nn.Flatten()
        
        # 3. CLASSIFIER WITH SOFTMAX
        classifier = nn.Sequential(OrderedDict([
                ('0', nn.Linear(in_features=self.flatten_nodes, 
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