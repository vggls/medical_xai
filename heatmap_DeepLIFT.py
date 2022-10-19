"""
Sources
 - Paper  : https://arxiv.org/pdf/1704.02685.pdf
            Includes links to video tutorial, ICML presentation and GitHub code
 - The code included in this file is based on the following Github tutorial : 
     https://github.com/pytorch/captum/blob/master/tutorials/CIFAR_TorchVision_Interpret.ipynb
 - Captum DeepLift API : https://captum.ai/api/deep_lift.html
"""

#############################################################################################################
#pip install captum

import numpy as np
from PIL import Image
from skimage.measure import block_reduce

import torchvision.transforms as transforms
from torch import nn

from captum.attr import DeepLift

#############################################################################################################

def ReLU_inplace_to_False(model):
    
    for layer in model._modules.values():
        
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)
        
    return model

class DeepLIFT_heatmap():
    
    def __init__(self, img_path, model):
        
        self.img_path = img_path                      # the image path
        self.img_size = 128                           # the resize size applied to the image
        self.normalization_mean = [0.5, 0.5, 0.5]     # mean of the image normalization transformation
        self.normalization_std = [0.5, 0.5, 0.5]      # std of the image normalization transformation
        self.model = model                            # the neural network model
        self.region = 16                              # the region side-length of the region-based heatmap. As per default values 
                                                      # Thus, as per default values heatmap.shape = (128/16, 128/16) = (8, 8)
        
    def heatmap(self):
        
        # Fails with resnet and inception; regardless of the check_ReLU_inplace method 
        # More info for resnet here --> https://github.com/pytorch/captum/issues/378

        self.model.zero_grad()

        tensor = self.transform()
        tensor.requires_grad = True
        
        # if architecture has ReLU then set inplace to False (ex. vgg, densenet, mobilenet or hand-written)
        # if no ReLU then do nothing (ex. inception, efficientnets, googlenet)
        if ('Inception' not in self.model.__class__.__name__) and \
           ('EfficientNet' not in self.model.__class__.__name__) and \
           ('GoogLeNet' not in self.model.__class__.__name__):
            print('ReLU path')
            self.model = ReLU_inplace_to_False(self.model)
        else:
            print('No ReLU path')
            pass
        
        attr_dl = DeepLift(self.model).attribute(tensor,
                                                 target=int(np.argmax(self.model(tensor).detach().numpy())))
        attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        attributions = np.mean(attr_dl, axis = -1)                                               # (128, 128)

        positive_attributions = np.where(attributions>0, attributions, 0)                        # ReLU
        heatmap = block_reduce(positive_attributions, (self.region,self.region), np.mean)        # AvgPooling

        regions_dict = {}
        for i in range(heatmap.shape[0]):
          for j in range(0, heatmap.shape[0]):
            regions_dict[i,j] = heatmap[i,j]
        regions = sorted(regions_dict.items(), key=lambda x: x[1], reverse=True)

        return attributions, heatmap, regions
    
    def transform(self):

        img_rgb = Image.open(self.img_path).convert('RGB').resize((self.img_size,self.img_size)) #PIL type img

        transformation = transforms.Compose([
            #transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std)
        ])

        img_tensor = transformation(img_rgb).unsqueeze(0)

        return img_tensor