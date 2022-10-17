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

from captum.attr import DeepLift

#############################################################################################################

class DeepLIFT_heatmap():

  def __init__(self, img_path, model):

    self.img_path = img_path                      # the image path
    self.img_size = 128                           # the resize size applied to the image
    self.normalization_mean = [0.5, 0.5, 0.5]     # mean of the image normalization transformation
    self.normalization_std = [0.5, 0.5, 0.5]      # std of the image normalization transformation
    self.model = model                            # the neural network model
    self.region = 16                              # the region side-length of the region-based heatmap. As per default values 
                                                  # Thus, as per default values heatmap.shape = (128/16, 128/16) = (8, 8)

  def transform(self):

    img_rgb = Image.open(self.img_path).convert('RGB').resize((self.img_size,self.img_size)) #PIL type img

    transformation = transforms.Compose([
        #transforms.Resize((self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std)
    ])

    img_tensor = transformation(img_rgb).unsqueeze(0)

    return img_tensor

  def heatmap(self):

    self.model.zero_grad()

    tensor = self.transform()
    tensor.requires_grad = True

    # set all inplace = False in the model Relu functions (otherwise it will yield error)
    nnparts = [self.model.features, self.model.classifier]
    for part in nnparts:
      for i, layer in enumerate(list(part)):
        string = part[i]
        if 'ReLU' in str(string):
          try:
            layer.inplace = False
          except:
            pass

    attributions = DeepLift(self.model).attribute(tensor,
                                             target=int(np.argmax(self.model(tensor).detach().numpy())))
    attributions = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attributions = np.mean(attributions, axis = -1) # np.mean over channels; converts 3-dim to 2-dim
    attributions = np.where(attributions>0, attributions, 0) # ReLU

    heatmap = block_reduce(attributions, (self.region,self.region), np.mean)

    regions_dict = {}
    for i in range(heatmap.shape[0]):
      for j in range(0, heatmap.shape[0]):
        regions_dict[i,j] = heatmap[i,j]
    regions = sorted(regions_dict.items(), key=lambda x: x[1], reverse=True)

    return attributions, heatmap, regions
