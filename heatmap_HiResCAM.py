"""
Sources
 - Paper  :  --
 - GitHub : https://github.com/jacobgil/pytorch-grad-cam
 - The code included in this file is based on the following Github tutorial : 
     https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Pixel%20Attribution%20for%20embeddings.ipynb    

"""
'''
Remark on the target_layer attribute :
        - It is the layer wrt which we compute the (high prob) class derivatives
        - Usually this will be the last convolutional layer in the model.
          In this case, as remarked in Github by the authors, some common choices can be:
          Resnet18 and 50: model.layer4
          VGG, densenet161: model.features[-1]
          mnasnet1_0: model.layers[-1]
          
Remark on the 'targets' parameter of the HiResCAM methdod :
     As per lines 105-110 from this code https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py
     as long as we are ok with getting only the heatmap of the highest prob category 
     we are ok with the below implementation and we do not have to set any value to this parameter (default is None)
'''

##############################################################################################
# pip install grad-cam

import numpy as np
import cv2
from skimage.measure import block_reduce

from torchvision import transforms

from pytorch_grad_cam import HiResCAM

##############################################################################################

class HiResCAM_heatmap():
    
    def __init__(self, img_path, model, target_layer):
        
        self.img_path = img_path                      # the image path
        self.img_size = 128                           # the resize size applied to the image
        self.normalization_mean = [0.5, 0.5, 0.5]     # mean of the image normalization transformation
        self.normalization_std = [0.5, 0.5, 0.5]      # std of the image normalization transformation
        self.model = model                            # the neural network model
        self.target_layer = target_layer              # (see detailed description above)
        self.region = 16                              # the region side-length of the region-based heatmap. As per default values 
                                                      # Thus, as per default values heatmap.shape = (128/16, 128/16) = (8, 8)


    def heatmap(self):

        img_tensor = self.preprocess() # tensor.Size(1,3,128,128)

        #self.model.eval()
        cam = HiResCAM(model=self.model,
                       target_layers=[self.target_layer],
                       use_cuda=False)

        attributions = cam(input_tensor=img_tensor)[0,:,:]                          # (128, 128)

        attributions = np.where(attributions>0, attributions, 0)                    # ReLU
        heatmap = block_reduce(attributions, (self.region,self.region), np.mean)    # AvgPooling

        # Use below code to plt.imshow() the heatmap with the values on the regions as text - Works ok
        #plt.matshow(heatmap)
        #for (x, y), value in np.ndenumerate(heatmap.transpose()):
        #    plt.text(x, y, f"{value:.2f}", va="center", ha="center")

        regions_dict = {}
        for i in range(heatmap.shape[0]):
          for j in range(0, heatmap.shape[0]):
            regions_dict[i,j] = heatmap[i,j]
        regions = sorted(regions_dict.items(), key=lambda x: x[1], reverse=True)

        return attributions, heatmap, regions


    def preprocess(self): 

        img_rgb = cv2.imread(self.img_path, 1)[:, :, ::-1]#cv2 loads BRG images. The [--] part converts the image to rbg. Equiv could have used np.flip(img, axis=-1) 
        img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size)) ## this is nd.array in range [0,255]

        #As per https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html 
        #the method "ToTensor" normalizes to [0,1] range
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std)
        ])

        img_tensor = transformation(img_rgb).unsqueeze(0)

        return img_tensor
