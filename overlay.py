"""
INTRODUCTION
    
    => Goal : The purpose of the file is to generate a super-imposed version of the original image by adding 
           a weighted version of the XAI algorithm heatmap. 
    
    => Source : We use the 'save_and_display_gradcam' method of the Grad-CAM keras implementation;
             as included in the following link https://keras.io/examples/vision/grad_cam/  .
             The computed images are of the form:
                            superimposed = image + alpha * jet_heatmap, where alpha in [0,1]   (*)
             See code below for a detailed construction of the jet_heatmap.

    => An alternative approach (not considered in this file) :
        Another way to combine the original image and the heatmap would be to take their weighted average as per :
                superimposed = (1 - image_weight) * heatmap + image_weight * img, where image_weight in [0,1]   (**)
        This approach in implemented in the Grad-CAM pytorch implementation and is included in the following link 
        https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/image.py .
        
        In this file (*) was chosen over (**) as we wanted to keep the original image unchanged.
        
    => Remark : 
        Note that despite both formulas are sourced from Grad-CAM implementations, they can obviously be applied to any
        kind of XAI algorithm.
"""

import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from torch import Tensor


def superimposed(tensor, heatmap, alpha):    
    
    '''
    Arguments
    tensor: a 3-dim torch tensor
    heatmap: a 2-dim np.array
    alpha: number in [0,1]
    
    Outputs
    img_array: the tensor converted to a numpy array
    superimposed_img: the overlay image, as per (*)
    '''
    
    assert Tensor.dim(tensor) == 3
    
    img_array = tensor.permute(1,2,0).detach().numpy()    

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return img_array, superimposed_img
