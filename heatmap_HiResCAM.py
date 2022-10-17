"""
Sources
 - Paper  :  --
 - GitHub : https://github.com/jacobgil/pytorch-grad-cam
 - The code included in this file is based on the following Github tutorial : 
     https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Pixel%20Attribution%20for%20embeddings.ipynb    

"""

##############################################################################################
# pip install grad-cam

import numpy as np
import cv2
from skimage.measure import block_reduce

from torchvision import transforms

from pytorch_grad_cam import HiResCAM

##############################################################################################

def hirescam_heatmap(img_path, model, target_layer, size):
    
    '''
    Arguments
    img_path: The path to the image 
    model: A pytorch neural network model
    target_layer: The layer we want to compute the visualization for
    size: The region side length of the region-based heatmap
    
    Outputs
    attributions: The pixel-level heatmap. It has shape (img_size, img_size).
    heatmap: The region-level heatmap. It has shaped (img_size//size, img_size//size)
    regions: The heatmap regions in descending order of importance
    '''
    
    '''
    Remark1: On the target_layer
        - It is the layer wrt which we compute the (high prob) class derivatives
        - Usually this will be the last convolutional layer in the model.
          In this case, as remarked in Github by the authors, some common choices can be:
          Resnet18 and 50: model.layer4
          VGG, densenet161: model.features[-1]
          mnasnet1_0: model.layers[-1]
        
    Remark2: On the 'targets' parameter
         As per lines 105-110 from this code https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py
         as long as we are ok with getting only the heatmap of the highest prob category 
         we are ok with the below implementation and we do not have to set any value to this parameter (default is None)
    '''

    img_tensor = preprocess(img_path) # tensor.Size(1,3,128,128)

    model.eval()
    cam = HiResCAM(model=model,
                   target_layers=[target_layer],
                   use_cuda=False)

    attributions = cam(input_tensor=img_tensor)[0,:,:]

    heatmap = block_reduce(attributions, (size,size), np.mean)

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


def preprocess(img_path): 
    
    size = 128

    img_rgb = cv2.imread(img_path, 1)[:, :, ::-1]# cv2 loads BRG images. The [--] part converts the image to rbg. Equiv could have used np.flip(img, axis=-1) 
    img_rgb = cv2.resize(img_rgb, (size, size)) ## this is nd.array in range [0,255]
    
    #As per https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html 
    #the method "ToTensor" normalizes to [0,1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img_rgb).unsqueeze(0)

    return img_tensor
