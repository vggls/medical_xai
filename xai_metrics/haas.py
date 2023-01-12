"""
Source 
    - Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759
"""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
import time

#-----------------------------------------------------------------------------------------

def Haas(dataset, model, cam_instance):
    
    '''
    Arguments
    dataset: data loaded via pytorch ImageFolder method. Normalized in [-1,1] (usually via transforms.Normalize w/ mean=std=[0.5, 0.5, 0.5])
    model: pytorch model
    cam_instance
         
    Outputs
    haas_score: as per formula (5) of the aforementioned paper
    images, ha_images, targets: summary of the correctly classified datapoints
    '''          
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
            
    y_pred =[]; ha_y_pred = []; targets= []
    
    t0 = time.time()
    
    print('No. of test images: ', len(dataset))
    
    for image, label in dataset:
        
        image = image.to(device)
        
        targets.append(int(label))
        
        y_pred.append(int(torch.argmax(model(image.unsqueeze(0)), dim=1).cpu().detach().numpy()))
                        
        assert (torch.max(image) <= 1) and (torch.min(image) >= -1) # as per authors' instructions
                    
        # get pixel attributions ('heatmap' as referenced in the paper)
        pixel_attributions = cam_instance(input_tensor=image.unsqueeze(0))[0,:,:]
    
        assert (np.max(pixel_attributions) <= 1) and (np.min(pixel_attributions) >= -1)
        
        # HA image construction - by the max/min definition values are in [-1,1]
        ha_image = HA_image(image, pixel_attributions)
        
        ha_y_pred.append(int(torch.argmax(model(ha_image.unsqueeze(0)), dim=1).cpu().detach().numpy()))
    
    t1 = time.time()
    
    print('Total time: {} secs'.format(t1-t0))
    
    score = round(accuracy_score(targets, y_pred), 2)
    ha_score = round(accuracy_score(targets, ha_y_pred), 2)
   
    print('Score over original images: ', score)
    print('Score over HA images: ', ha_score)
  
    haas_score = round(ha_score / score, 2)
    
    return haas_score, targets, y_pred, ha_y_pred


def HA_image(image, attributions):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    channel_scalar = torch.from_numpy(attributions).to(device) + \
        torch.ones(attributions.shape[0], attributions.shape[1]).to(device)  # 3dim torch tensor
    for channel in range(image.shape[0]):
        image[channel,:,:] = image[channel,:,:] * channel_scalar
    ones = torch.ones(image.shape[0], image.shape[1], image.shape[2]).to(device)
    ha_image = torch.maximum((-1)*ones, torch.minimum(ones, image))  # 3dim torch tensor
    assert image.shape == ha_image.shape
    
    return ha_image
