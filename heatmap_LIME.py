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

import torch
from torchvision import transforms
import torch.nn.functional as F

from lime import lime_image

#########################################################################################################

class LIME_heatmap():
    
    def __init__(self, img_path, model):
        
        self.img_path = img_path                      # the image path
        self.img_size = 128                           # the resize size applied to the image
        self.normalization_mean = [0.5, 0.5, 0.5]     # mean of the image normalization transformation
        self.normalization_std = [0.5, 0.5, 0.5]      # std of the image normalization transformation
        self.model = model                            # the neural network model
        self.region = 16                              # the region side-length of the region-based heatmap. As per default values 
                                                      # Thus, as per default values heatmap.shape = (128/16, 128/16) = (8, 8)

    def heatmap(self):
        
        img_rgb = Image.open(self.img_path).convert('RGB').resize((self.img_size,self.img_size))
        image = np.array(img_rgb)
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image,           
                                             classifier_fn=self.batch_predict,
                                             top_labels=1, 
                                             hide_color=0, 
                                             num_samples=1000)  # number of images that will be sent to classification function
        
        exp_class = explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[exp_class])
        attributions = np.vectorize(dict_heatmap.get)(explanation.segments)    # (128,128)
        
        positive_attributions = np.where(attributions>0, attributions, 0)                     # ReLU
        heatmap = block_reduce(positive_attributions, (self.region,self.region), np.mean)     # AvgPooling
        
        regions_dict = {}
        for i in range(heatmap.shape[0]):
          for j in range(0, heatmap.shape[0]):
            regions_dict[i,j] = heatmap[i,j]
        regions = sorted(regions_dict.items(), key=lambda x: x[1], reverse=True)
        
        return attributions, heatmap, regions

    def batch_predict(self, images):

        '''
        Description : Classifier prediction probability function, 
                      which takes a numpy array and outputs prediction probabilities
        
        '''        

        stacked = np.stack(images, axis=0)

        self.model.eval()
        batch = torch.stack(tuple(self.transform(i) for i in stacked), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch = batch.to(device)

        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)

        return probs.detach().cpu().numpy()


    def transform(self, image):

        # 'img' can be either PIL.Image.Image or np.array of the Image shaped as (width, height, channels)

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std)
        ])

        img_tensor = transformation(image)

        return img_tensor
