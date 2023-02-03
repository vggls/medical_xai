All models are customized according to the following architecture: CNN part (last layer is Conv2d layer) - Flatten - Raw class scores

In this file there is a detailed description of the class attributes included in the .py files.

    --> (resnet.py) type_: one of '34', '50'

    --> no_of_classes: integer

    --> trainable_feature_layers / trainable_layers:
                                  This attribute is either 'None' (default) OR a list of indices where each index corresponds to a model's feature layer.
                                  We note that the classifier / fully connected part of the model is always trainable.
                                                                    
                                  ------ResNets------    |   ------DenseNets------
                                  LAYER        INDEX     |   LAYER           INDEX
                                  conv1     :    0       |   conv0       :     0
                                  bn1       :    1       |   norm0       :     1
                                  relu      :    2       |   relu0       :     2
                                  maxpool   :    3       |   pool0       :     3
                                  layer1    :    4       |   denseblock1 :     4
                                  layer2    :    5       |   transition1 :     5
                                  layer3    :    6       |   denseblock2 :     6
                                  layer4    :    7       |   transition2 :     7
                                                         |   denseblock3 :     8
                                                         |   transition3 :     9
                                                         |   denseblock4 :     10
                                                         |   norm5       :     11
                                                         
                                  ------------VGG19------------
                                  Indices range between 0 and 34. 
                                  Run below code to see the indices of the feature layers:
                                     from torchvision import models
                                     vgg = models.vgg19(pretrained=True)
                                     vgg.features
                                  
                                  (Note that ReLU and MaxPool layers do not have trainable params)
 

