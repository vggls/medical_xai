"""
Sources
 - Paper  : https://arxiv.org/pdf/1602.04938.pdf
 - GitHub : https://github.com/marcotcr/lime
 - The code included in this file is based on the following Github tutorial : 
     https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb    
 - The documentation of the "lime_image" module imported below for the heatmap purposes :   
     https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image
"""

#########################################################################################################
# pip install lime

from PIL import Image
import numpy as np
from skimage.measure import block_reduce
import functools

import torch
from torchvision import transforms
import torch.nn.functional as F

from lime import lime_image

#########################################################################################################

def lime_heatmap(img_path, model, size):
    
    '''
    Arguments
    img_path: The path to the image 
    model: A pytorch neural network model
    size: The region side length of the region-based heatmap
    
    Outputs
    attributions: The pixel-level heatmap. It has shape (img_size, img_size).
    heatmap: The region-level heatmap. It has shaped (img_size//size, img_size//size)
    regions: The heatmap regions in descending order of importance
    '''
    
    img_size = 128 # All images are resized at this shape. If necessary put it as parameter
    img_rgb = Image.open(img_path).convert('RGB').resize((img_size,img_size)) #PIL type img
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(img_rgb),             
                                         classifier_fn=functools.partial(batch_predict, model),
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=1000)  # number of images that will be sent to classification function
    
    exp_class = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[exp_class])
    attributions = np.vectorize(dict_heatmap.get)(explanation.segments) #2-dim, (img_size, img_size)
    
    heatmap = block_reduce(attributions, (size,size), np.mean)
    
    regions_dict = {}
    for i in range(heatmap.shape[0]):
      for j in range(0, heatmap.shape[0]):
        regions_dict[i,j] = heatmap[i,j]
    regions = sorted(regions_dict.items(), key=lambda x: x[1], reverse=True)

    return attributions, heatmap, regions


def batch_predict(model, images):
    
    '''
    Description : Classifier prediction probability function, 
                  which takes a numpy array and outputs prediction probabilities
    '''
    
    model.eval()
    batch = torch.stack(tuple(transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    
    return probs.detach().cpu().numpy()


def transform(img_rgb):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img_rgb)

    return img_tensor