
"""
ATTRIBUTIONS

    => HiResCAM algorithm
        a) GitHub: https://github.com/jacobgil/pytorch-grad-cam
        b) example code:
             # pip install grad-cam
             from pytorch_grad_cam import HiResCAM
             cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=False)
             pixel_attributions = cam(input_tensor=tensor.unsqueeze(0))[0,:,:]  # numpy.ndarray, 2-dim, values in [0,1]
        c) Remarks on the target_layer attribute:
            - It is the layer wrt which we compute the class derivatives
            - Usually this will be the last convolutional layer in the model.
              In this case, as remarked in Github by the authors, some common choices can be:
              Resnet18 and 50: [model.layer4]
              vgg16, vgg19, densenet161, densenet201, efficientnets, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large: [model.features[-1]]
              mnasnet1_0: model.layers[-1]     
        d) VERY IMPORTANT: In case the target layer was frozen for the training phase, we should unfreeze it in order to pass the model as HiResCAM argument.
                           Otherwise, TypeError occurs.  
        d) Remark on the 'targets' parameter of the HiResCAM methdod:
             As per lines 105-110 of https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py
             above code returns only the heatmap of the highest prob category (targets=None by default).
             BUT in order to get the heatmaps from the rest classes (if necessary) one of the following Target classes should be used
             https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/model_targets.py
    
"""

import numpy as np
from skimage.measure import block_reduce

def heatmap(pixel_attributions, region_size):
            
    assert pixel_attributions.shape[0]%region_size == 0
    
    heat_map = block_reduce(pixel_attributions, (region_size, region_size), np.mean)  # AvgPooling
    
    regions_dict = {}
    for i in range(heat_map.shape[0]):
        for j in range(0, heat_map.shape[0]):
            regions_dict[i,j] = heat_map[i,j]
    regions = sorted(regions_dict.items(), key=lambda x: x[1], reverse=True)
    
    return heat_map, regions
