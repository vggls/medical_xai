from torchvision import models
from torch import nn

class VGG():
    
    '''
    ATTRIBUTES: see README_attributes.md
    '''
    
    def __init__(self, type_, no_of_classes, custom_classifier, trainable_feature_layers=None):
        
        self.type_ = type_
        self.no_of_classes = no_of_classes
        self.trainable_feature_layers = trainable_feature_layers
        self.custom_classifier = custom_classifier
        
        if self.type_ == '16':
            self.model = models.vgg16(pretrained=True)
        elif self.type_ == '19':
            self.model = models.vgg19(pretrained=True)
        
        # CLASSIFIER
        assert self.custom_classifier[-2].out_features == self.no_of_classes
        assert type(custom_classifier[-1]) == nn.modules.activation.Softmax
        self.model.classifier = self.custom_classifier
        
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
        print('No. of trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def unfreeze(self):
        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = True