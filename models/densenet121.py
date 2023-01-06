'''
Source: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

The code for _bn_function_factory method and _DenseLayer, _DenseBlock and _Transition classes
is copied from the above source.

The DenseNet121 class is the source class with the following modifications in the network structure:
    1. The last BatchNorm2d feature layer is removed
    2. The GAP layer is removed (a Flatten layer connects the last Conv2d layer with the raw class scores instead)

The _DenseNet121_ class is a custom written class which defines a DenseNet121 instance and loads the pretrained 
feature weights. In addition it allows the user to choose the feature layers they want to train.
For the class attributes refer to README_attributes.md

ex. code to instantiate a model with the above structure
model = _DenseNet121_(num_classes=4,
                      trainable_feature_layers = [9,10])
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" `_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_featuremaps (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" `_
    """

    def __init__(self, 
                 num_classes,
                 growth_rate=32, 
                 block_config=(6, 12, 24, 16),
                 num_init_featuremaps=64, 
                 bn_size=4, 
                 drop_rate=0, 
                 memory_efficient=False,
                 grayscale=False):

        super(DenseNet121, self).__init__()

        # First convolution
        if grayscale:
            in_channels=1
        else:
            in_channels=3
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=in_channels, out_channels=num_init_featuremaps,
                                kernel_size=7, stride=2,
                                padding=3, bias=False)), # bias is redundant when using batchnorm
            ('norm0', nn.BatchNorm2d(num_features=num_init_featuremaps)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_featuremaps
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        #self.features.add_module('norm5', nn.BatchNorm2d(num_features))  # <-- NEW

        # Linear layer
        #self.classifier = nn.Linear(num_features, num_classes)
        self.classifier = nn.Linear(1024*7*7, self.num_classes)   # <-- NEW

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = F.adaptive_avg_pool2d(out, (1, 1)) # <-- NEW
        out = torch.flatten(out, 1) # Note that we keep this value at dim=1 as data is passed in batches
        logits = self.classifier(out)
        #probas = F.softmax(logits, dim=1)  # <-- NEW
        return logits#, probas
    

class _DenseNet121_():
    
    def __init__(self,
                 no_of_classes,
                 trainable_feature_layers):
    
        self.no_of_classes = no_of_classes
        self.trainable_feature_layers = trainable_feature_layers

        self.model = DenseNet121(num_classes=self.no_of_classes)

        # download pretrained weights
        url="https://download.pytorch.org/models/densenet121-a639ec97.pth"
        weights = torch.hub.load_state_dict_from_url(url)

        # Below code corresponds to the _load_state_dict method of the official implementation code
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        new_keys = []
        for key in list(weights.keys()):

            if 'norm5.' in key:
                continue

            if 'features.' in key:
                new_key = key.replace('features.', '')
                if 'norm.1' in new_key:
                    new_key = new_key.replace('norm.1', 'norm1')
                    new_keys.append(new_key)
                elif 'norm.2' in new_key:
                    new_key = new_key.replace('norm.2', 'norm2')
                    new_keys.append(new_key)
                elif 'conv.1' in new_key:
                    new_key = new_key.replace('conv.1', 'conv1')
                    new_keys.append(new_key)
                elif 'conv.2' in new_key:
                    new_key = new_key.replace('conv.2', 'conv2')
                    new_keys.append(new_key)
                elif 'relu.1' in new_key:
                    new_key = new_key.replace('relu.1', 'relu1')
                    new_keys.append(new_key)
                elif 'relu.2' in new_key:
                    new_key = new_key.replace('relu.2', 'relu2')
                    new_keys.append(new_key)
                else:
                    new_keys.append(new_key)

        # does not contain the last batch norm and the classifier weights
        adjusted_feature_weights = OrderedDict(zip(new_keys, list(weights.values())))
        #print(len(adjusted_feature_weights.keys()))

        # load pretrained feature weights
        self.model.features.load_state_dict(adjusted_feature_weights)

            # trainable feature layers
        if self.trainable_feature_layers==None:
            self.freeze = self.model.features
        else:
            len_ = len(self.model.features) #11
            assert all(x in range(len_) for x in self.trainable_feature_layers)
            self.freeze = [self.model.features[j] for j in range(len_) if j not in self.trainable_feature_layers]

        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = False
    
    def trainable_params(self):
        print('No. of trainable feature params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))