"""
Source 
    - Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759
"""

import numpy as np
from sklearn.metrics import accuracy_score #to load more; for instance recall for medical, f1 for imbalanced 
import torch

from heatmaps import hirescam, deeplift, lime     

#-----------------------------------------------------------------------------------------

def Haas(dataloader, model, xai_algorithm, target_layer=None):
    
    '''
    Arguments
    dataloader: data loaded via Dataloaders. Normalized in [-1,1] (usually via transforms.Normalize w/ mean=std=[0.5, 0.5, 0.5])
    model: pytorch model
    xai_algorithm: one of 'hirescam', 'deeplift' and 'lime'
    target_layer: the layer wrt which the gradients are computed in 'hirescam'
         
    Outputs
    haas_score: as per formula (5) of the aforementioned paper
    images, ha_images, targets: summary of the correctly classified datapoints
    '''          
    
    assert xai_algorithm in ['hirescam', 'deeplift', 'lime']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
            
    y_pred =[]; ha_y_pred = []; targets= [];
    
    for batch_images, batch_labels in dataloader:
        for image, real_label in zip(batch_images, batch_labels):
            image, real_label = image.to(device), real_label.to(device)
            
            targets.append(int(real_label.cpu().detach().numpy()))
            
            y_pred.append(int(torch.argmax(model(image.unsqueeze(0)), dim=1).cpu().detach().numpy()))
                            
            assert (torch.max(image) <= 1) and (torch.min(image) >= -1) # as per authors' instructions
                        
            # get pixel attributions ('heatmap' as referenced in the paper)
            if xai_algorithm == 'hirescam':
                assert target_layer != None
                attributions = hirescam(image, model, target_layer)
            elif xai_algorithm == 'deeplift':
                attributions = deeplift(image, model)
            elif xai_algorithm == 'lime':
                attributions = lime(image, model)
            assert (np.max(attributions) <= 1) and (np.min(attributions) >= -1)
            
            # HA image construction - by the max/min definition values are in [-1,1]
            ha_image = HA_image(image, attributions)
            
            ha_y_pred.append(int(torch.argmax(model(ha_image.unsqueeze(0)), dim=1).cpu().detach().numpy()))
                    
    # metric for original images
    score = round(accuracy_score(targets, y_pred), 2)
    print('Score over original images: ', score)

    # metric for HA images
    ha_score = round(accuracy_score(targets, ha_y_pred), 2)
    print('Score over HA images: ', ha_score)
  
    haas_score = round(ha_score / score, 2)
    
    return haas_score

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