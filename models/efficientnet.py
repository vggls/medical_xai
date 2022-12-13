from collections import OrderedDict
from torchvision import models
from torch import nn

class EfficientNet():
    
    '''
    ATTRIBUTES: see README_attributes.md
    '''
    
    def __init__(self, type_, 
                 no_of_classes, 
                 trainable_feature_layers=None, 
                 flatten=True):
        
        self.type_ = type_
        self.no_of_classes = no_of_classes
        self.trainable_feature_layers = trainable_feature_layers
        self.flatten = flatten
        
        if self.type_ == 'b1':
            self.model = models.efficientnet_b1(pretrained=True)
            self.flatten_nodes = 1280 * 7 * 7; self.dropout_prob = 0.2
        elif self.type_ == 'b2':
            self.model = models.efficientnet_b2(pretrained=True)
            self.flatten_nodes = 1408 * 7 * 7; self.dropout_prob = 0.3
        elif self.type_ == 'b3':
            self.model = models.efficientnet_b3(pretrained=True)
            self.flatten_nodes = 1536 * 7 * 7; self.dropout_prob = 0.3
        elif self.type_ == 'b4':
            self.model = models.efficientnet_b4(pretrained=True)
            self.flatten_nodes = 1792 * 7 * 7; self.dropout_prob = 0.4
        elif self.type_ == 'b5':
            self.model = models.efficientnet_b5(pretrained=True)
            self.flatten_nodes = 2048 * 7 * 7; self.dropout_prob = 0.4
        elif self.type_ == 'b6':
            self.model = models.efficientnet_b6(pretrained=True)
            self.flatten_nodes = 2304 * 7 * 7; self.dropout_prob = 0.5
        elif self.type_ == 'b7':
            self.model = models.efficientnet_b7(pretrained=True)
            self.flatten_nodes = 2560 * 7 * 7; self.dropout_prob = 0.5
        
        # REPLACE GAP WITH FLATTEN
        if self.flatten == True:
            self.model.avgpool = nn.Flatten() # haven't changed layer name though - to do
        
        # CLASSIFIER WITH SOFTMAX
        if self.flatten == False:
            num_filters = self.model.classifier[-1].in_features
        elif self.flatten == True:
            num_filters = self.flatten_nodes
        classifier = nn.Sequential(OrderedDict([
            ('0', nn.Dropout(p=self.dropout_prob, inplace=True)),
            ('1', nn.Linear(in_features=num_filters, 
                            out_features=self.no_of_classes)),
            ('2', nn.Softmax(dim=1))
        ]))
        self.model.classifier = classifier
        
        # LAYERS TO FREEZE DURING TRAINING
        if self.trainable_feature_layers==None:
            self.freeze = self.model.features
        else:
            len_ = len(self.model.features)
            assert all(x in range(len_) for x in self.trainable_feature_layers)
            self.freeze = [self.model.features[j] for j in range(len_) if j not in self.trainable_feature_layers]
            
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