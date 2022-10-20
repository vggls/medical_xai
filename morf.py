"""
Sources
    - Paper : https://arxiv.org/pdf/1509.06321.pdf 
"""

import numpy as np
import matplotlib.pyplot as plt
#from torch import nn
import torch.nn.functional as F
import torch

class MoRF():

  # At single tensor (image) level. It considers the top predicted class before the perturbations.

  def __init__(self, tensor, heatmap_regions, model, noise):

      self.tensor = tensor                    # tensor (Note that after ToTensor+Normalize values will be in [-1, 1], but that is ok for the network)
      self.heatmap_regions = heatmap_regions  # the image heatmap regions in order of importance (resulting from a XAI algorithm)
      self.model = model                      # a neural network model
      self.noise = noise                      # tensor ex. uniform_noise = (torch.min(tensor) - torch.max(tensor)) * torch.rand(shape) + torch.max(tensor)
                                              #        ex. normal_noise = torch.normal(mean=0, std=1, size) + needs to fix values below -1 and above 1 (if any)
        
      self.perturbation_size = int(tensor.shape[3]/np.sqrt(len(self.heatmap_regions)))       #  the size of the image region that will be perturbed

      isinstance(self.tensor, torch.Tensor)
      isinstance(self.noise, torch.Tensor)
      assert( noise.shape == (3, self.perturbation_size, self.perturbation_size) ) # assert that noise shape is the same shape as the heatmap regions

  def perturbations(self, plot_probabilities=False):

      # list to store all predicted probabilities of the top predicted class
      perturbations = []

      # top predicted class info
      predict = self.model(self.tensor)
      probs = F.softmax(predict, dim=1)
      index = int(np.argmax(probs.detach().numpy(), axis = 1)) # index of the top predicted class
      prob = round(float(probs[0,index]), 3)                   # probability of the top predicted class
      perturbations.append(prob)

      image = self.tensor.detach().clone()                     # do this in order to have access to the initial tensor after the perturbations

      for step in range(0, len(self.heatmap_regions)):

        # get region row and column
        region_row = self.heatmap_regions[step][0][0]
        region_column = self.heatmap_regions[step][0][1]

        # apply noise to tensor region
        image[:,:3,region_row*self.perturbation_size:(region_row+1)*self.perturbation_size, \
               region_column*self.perturbation_size:(region_column+1)*self.perturbation_size] = self.noise

        # make prediction and get value via the index used above (see prob variable line). this value will be the class predicted probability
        perturbed_prob = round(float(F.softmax(self.model(image), dim=1)[0,index]), 3)
        
        # append the value to perturbations
        perturbations.append(perturbed_prob)

      if plot_probabilities==True:
          plt.plot(range(0,len(perturbations)), perturbations)
          plt.xlabel('Perturbation steps')
          plt.ylabel('Predicted probability')
          plt.show()

      return perturbations

  def aopc(self, plot_morf_curve=False):

      probabilities = self.perturbations()

      unpert_prob = [probabilities[0] for _ in range(len(probabilities))]
      differences = [a - b for a, b in zip(unpert_prob, probabilities)]

      if plot_morf_curve==True:
          plt.plot(range(0,len(probabilities)), differences)
          plt.title('MoRF curve')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Probabilities difference')
          plt.show()

      return sum(differences)
