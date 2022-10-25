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
            
    images = []; ha_images = []; targets = []
    
    for batch_images, batch_labels in dataloader:
        for image, label in zip(batch_images, batch_labels):
            
            real_label = int(label.detach().numpy())
            pred_label = int(torch.argmax(model(image.unsqueeze(0)), dim=1).detach().numpy())
    
            if real_label == pred_label:
                
                assert (torch.max(image) <= 1) and (torch.min(image) >= -1)
                
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
                channel_scalar = torch.from_numpy(attributions) + \
                                torch.ones(attributions.shape[0], attributions.shape[1])  # 3dim torch tensor
                for channel in range(image.shape[0]):
                    image[channel,:,:] = image[channel,:,:] * channel_scalar
                ones = torch.ones(image.shape[0], image.shape[1], image.shape[2])
                ha_image = torch.maximum((-1)*ones, torch.minimum(ones, image))  # 3dim torch tensor
                assert image.shape == ha_image.shape
                
                # append - images, ha_images and targets lists contain all the info of the correctly labelled images
                images.append(image); ha_images.append(ha_image); targets.append(label)
                
    # stack list images in tensors batch for model prediction
    test_tensors = torch.stack(images, dim=0)
    test_ha_tensors = torch.stack(ha_images, dim=0)
    
    # original images
    outputs = model(test_tensors)
    preds = torch.argmax(outputs, dim=1)
    y_pred = list(preds.detach().numpy())
    score = accuracy_score(targets, y_pred)
    
    # ha images
    #ha_y_pred = []
    ha_outputs = model(test_ha_tensors)
    ha_preds = torch.argmax(ha_outputs, dim=1)
    ha_y_pred = list(ha_preds.detach().numpy())
    ha_score = accuracy_score(targets, ha_y_pred)
  
    haas_score = round((ha_score/score),2)
    
    return haas_score, images, ha_images, targets