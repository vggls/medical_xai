All models are customized according to the following architecture: CNN part (last layer is Conv2d layer) - Flatten - Raw class scores

In this file there is a detailed description of the class attributes included in the .py files.

    --> (resnet.py) type_: one of '34', '50'

    --> no_of_classes: integer

    --> trainable_feature_layers / trainable_layers:
                                  This attribute is either 'None' (default) OR a list of indices where each index corresponds to a model's feature layer.
                                  We note that the classifier / fully connected part of the model is always trainable.
                                                                    
                                  ------ResNets--------  
                                  LAYER        INDEX
                                  conv1     :    0
                                  bn1       :    1
                                  relu      :    2
                                  maxpool   :    3
                                  layer1    :    4    
                                  layer2    :    5
                                  layer3    :    6
                                  layer4    :    7
                                  
                                  ------VGG19------
                                  Indices range between 0 and 34. 
                                  Run below code to see the indices of the feature layers:
                                     from torchvision import models
                                     vgg = models.vgg19(pretrained=True)
                                     vgg.features
                                  
                                  (Note that ReLU and MaxPool layers do not have trainable params)
 

