from torchvision import models
from torch import nn

class VGG():
    
    '''
    ATTRIBUTES
    
    type_: '16' or '19' to load pretrained vgg16 or vgg19 respectively
    
    no_of_classes: integer
    
    trainable_feature_layers: The vgg16 consists of 31 feature layers, indexed from 0 to 30 
                        while the vgg19 of 37 indexed from 0 to 36.
                        This attribute is either 'None' (by default) if we want to freeze all the layers during training
                        OR a list of the feature layer indices that we want to train.
                        
    custom_classifier: custom written classifier
            REMARK: Note that the main number of trainable parameters in the vgg architectures is in the classifier part, 
                    which is the same for both architectures. It consists of 123.6M parameters.
                    Since they are too many to handle, we consider using custom written classifier instead.
            example: custom_classifier = nn.Sequential(OrderedDict([
                                          ('0', nn.Linear(in_features=1024, out_features=256, bias=True)),
                                          ('1', nn.Linear(in_features=256, out_features=no_of_classes, bias=True)),
                                          ('2', nn.Softmax(dim=1))
                                        ]))
    '''
    
    def __init__(self, type_, no_of_classes, custom_classifier, trainable_feature_layers=None):
        
        self.type_ = type_
        self.no_of_classes = no_of_classes
        self.trainable_feature_layers = trainable_feature_layers
        self.custom_classifier = custom_classifier
        
        if self.type_ == '16':
            self.model = models.vgg16(pretrained=True)
        elif self.type == '19':
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