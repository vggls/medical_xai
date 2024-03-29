'''
Calculating the pixel_attributions for the HiResCAM method:
        a) example code:
             pip install grad-cam # see https://github.com/jacobgil/pytorch-grad-cam
             from pytorch_grad_cam import HiResCAM
             cam = HiResCAM(model=model, 
                            target_layers=target_layers, 
                            use_cuda=True)
             #targets = [ClassifierOutputTarget(class_of_interest_as_integer))]
             pixel_attributions = cam(input_tensor=tensor.unsqueeze(0), 
                                        targets=None )[0,:,:]  # numpy.ndarray, 2-dim, values in [0,1]
        b) on the target_layer attribute of the HiResCAM instance:
            - It is the layer wrt which we compute the class derivatives
            - Usually this will be the last convolutional layer in the model.
              In this case, as remarked in Github by the authors, some common choices can be:
              Resnet18 and 50: [model.layer4[-1]]
              vgg16, vgg19, densenet161, densenet201, efficientnets, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large: [model.features[-1]]
              mnasnet1_0: model.layers[-1]     
        c) VERY IMPORTANT: In case the target layer was frozen for the training phase, we should unfreeze it in order to pass the model as HiResCAM argument.
                           Otherwise, TypeError occurs.  
        d) Remark on the 'targets' parameter:
             As per lines 105-110 of https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py
             above code returns only the heatmap of the highest prob category (targets=None by default).
             BUT in order to get the heatmaps from the rest classes (if necessary) one of the following Target classes should be used
             https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/model_targets.py
    
'''

import numpy as np
from skimage.measure import block_reduce

def Heatmap(pixel_attributions, region_size):
            
    #assert pixel_attributions.shape[0]%region_size == 0
    
    heatmap = block_reduce(pixel_attributions, (region_size, region_size), np.mean)  # AvgPooling
    
    regions_dict = {}
    for i in range(heatmap.shape[0]):
        for j in range(0, heatmap.shape[0]):
            regions_dict[i,j] = heatmap[i,j]
    heatmap_regions = sorted(regions_dict.items(), key=lambda x: x[1], reverse=True)
    
    return heatmap, heatmap_regions
