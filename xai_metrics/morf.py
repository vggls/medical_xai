"""
Sources: https://arxiv.org/pdf/1509.06321.pdf 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import gc
import pandas as pd
import plotly.express as px

from heatmap import Heatmap

#-------Image Level class---------------------------------------------------------------------------------------------------

class MoRF():

    def __init__(self,
                 tensor,
                 heatmap_regions,
                 model):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tensor = tensor.to(self.device)
        self.heatmap_regions = heatmap_regions
        self.model = model.to(self.device)
    
        if self.tensor.shape[3]%np.sqrt(len(self.heatmap_regions)) == 0:
            self.perturbation_size = int(self.tensor.shape[3]/np.sqrt(len(self.heatmap_regions)))
        else:
            self.perturbation_size = int(self.tensor.shape[3]/np.sqrt(len(self.heatmap_regions)))+1


  def perturbations(self, 
                    plot_morf_curve=False):

      softmax = nn.Softmax(dim=1)
      perturbations = []
      
      with torch.no_grad():
          
          raw_scores = self.model(self.tensor)
          probs = softmax(raw_scores)
          index = torch.argmax(probs) #of top predicted class
          class_prob = round(float(probs[0,index]), 3)
          perturbations.append(class_prob)
    
          noise = torch.rand(3, self.tensor.shape[2], self.tensor.shape[3]).to(self.device) * \
                   (torch.max(self.tensor) - torch.min(self.tensor)) + torch.min(self.tensor)
    
          for region in self.heatmap_regions:
              
              r = region[0][0] #region row
              c = region[0][1] #region column
              r_pixel_end = min((r + 1) * self.perturbation_size, self.tensor.shape[2])
              c_pixel_end = min((c + 1) * self.perturbation_size, self.tensor.shape[3])
    
              self.tensor[:, :, r*self.perturbation_size:r_pixel_end, c*self.perturbation_size:c_pixel_end] = \
                noise[:, r*self.perturbation_size:r_pixel_end, c*self.perturbation_size:c_pixel_end]

              perturbed_probs = softmax(self.model(self.tensor))
              class_perturbed_prob = round(float(perturbed_probs[0, index]), 3)
              perturbations.append(class_perturbed_prob)
    
      if plot_morf_curve==True:
          # A good heatmap results in a large area over the morf (perturbation) curve. This is where the 'AOPC' name comes from.
          # This area is controlled by the sum of differences f(x_0) - f(x_step). This is why we compute them in the 'aopc' method below.
          plt.plot(range(0,len(perturbations)), perturbations)
          plt.title('MoRF Perturbation Curve')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Predicted probability')
          plt.show()

      return perturbations

  def aopc(self, plot_cumulative_differences=False):

      probabilities = self.perturbations()
      differences = [probabilities[0] - probabilities[i] for i in range(len(probabilities))]
      L = len(self.heatmap_regions)
      score = (1/(L+1)) * sum(differences)

      if plot_cumulative_differences==True:
          # The area below this curve is the AOPC score.
          sum_of_differences = np.cumsum(differences)   #cumulative sum of differences
          plt.plot(range(0,len(probabilities)), sum_of_differences)
          plt.title('Cumulative differences')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Sum of differences')
          plt.show()

      return differences, round(score, 3)

#------Dataset Level methods----------------------------------------------------------------------------------------------------

def AOPC_Dataset(dataset,
                 model,
                 region_size,
                 cam_instance):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tensor, label = dataset[0]
    tensor = tensor.to(device)
    img_size = tensor.shape[2]
    if img_size%region_size==0:
        L = (img_size//region_size)**2
    else:
        L = (img_size//region_size + 1)**2
    differences = [0] * (L+1)

    scores = []

    total_time = 0
    no_of_correctly_classified = 0

    for image, label in dataset:
        image = image.to(device)

        real_label = int(label)#int(label.cpu().detach().numpy())
        pred_label = int(torch.argmax(model(image.unsqueeze(0)), dim=1).cpu().detach().numpy())

        if real_label == pred_label:

            no_of_correctly_classified += 1

            t0 = time.time()

            pixel_attributions = cam_instance(input_tensor=image.unsqueeze(0))[0,:,:]

            _, heatmap_regions = Heatmap(pixel_attributions, region_size)

            img_diffs, img_score = MoRF(image.unsqueeze(0), heatmap_regions, model).aopc()

            t1 = time.time()

            total_time += (t1-t0)

            differences = [(i+j) for (i,j) in zip(differences, img_diffs)]      #for plot method below
            scores.append(img_score)                                            #for total aopc score

    print('Total time: {} secs'.format(total_time))
    print('No of correctly classified images: {}/{}'.format(no_of_correctly_classified, len(dataset)))
    print('Avg secs per image: ', round(total_time/no_of_correctly_classified, 2))

    aopc = round(sum(scores)/len(scores), 3)

    return differences, scores, aopc


def plot_aopc_per_step(differences, no_of_correctly_classified, plot=False, plot_title=None):
        
    cumulative = list(np.cumsum(differences))
    
    cumulative = [x / no_of_correctly_classified for x in cumulative]  # normalization
    
    #cumulative = [x / len(differences) for x in cumulative]  # normalization
    cumulative = [x / (j+1) for (x,j) in zip(cumulative, range(len(differences)))] # normalization; might affect increasing line a bit
    
    aopc = [round(x, 2) for x in cumulative]

    data = {'steps': [i for i in range(len(cumulative))], 'AOPC': aopc}
    df = pd.DataFrame.from_dict(data)
    
    if plot:
        fig = px.line(df, x='steps', y='AOPC', height=400, width=600, title=plot_title)
        fig.show()
    
    return df
