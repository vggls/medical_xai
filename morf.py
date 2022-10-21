"""
Sources
    - Paper : https://arxiv.org/pdf/1509.06321.pdf 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

class MoRF():

  # at single tensor (image) level

  def __init__(self, tensor, heatmap_regions, model, noise):

      self.tensor = tensor                    # tensor (1,c,h,w) (Note that after ToTensor+Normalize values will be in [-1, 1], but that is ok for the network)
      self.heatmap_regions = heatmap_regions  # the image heatmap regions in order of importance (resulting from a XAI algorithm)
      self.model = model                      # a neural network model
      self.noise = noise                      # tensor ex. uniform_noise = (torch.min(tensor) - torch.max(tensor)) * torch.rand(shape) + torch.max(tensor)
                                              #        ex. normal_noise = torch.normal(mean=0, std=1, size) + needs to fix values below -1 and above 1 (if any)
        
      self.perturbation_size = int(tensor.shape[3]/np.sqrt(len(self.heatmap_regions)))       #  the size of the image region that will be perturbed

      isinstance(self.tensor, torch.Tensor)
      isinstance(self.noise, torch.Tensor)
      assert( noise.shape == (3, self.perturbation_size, self.perturbation_size) ) # assert that noise shape is the same shape as the heatmap regions

  def perturbations(self, plot_morf_curve=False):

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

      if plot_morf_curve==True:
          # A good heatmap results in a large area over the morf (perturbation) curve. This is where the 'AOPC' name comes from.
          # Note that this area is controlled by the sum of differences f(x_0) - f(x_step). This is why we compute them in the 'aopc' method.
          plt.plot(range(0,len(perturbations)), perturbations)
          plt.title('MoRF Perturbation Curve')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Predicted probability')
          plt.show()

      return perturbations

  def aopc(self, plot_differences=False, plot_accumulative_differences=False):

      probabilities = self.perturbations()

      unpert_prob = [probabilities[0] for _ in range(len(probabilities))]
      differences = [a - b for a, b in zip(unpert_prob, probabilities)]

      if plot_differences==True:
          plt.plot(range(0,len(probabilities)), differences)
          plt.title('Differences: f(x_0) - f(x_step)')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Probabilities difference')
          plt.show()

      if plot_accumulative_differences==True:
          # Note that the area below this curve is the AOPC score.
          sum_of_differences = np.cumsum(differences)   #cumulative sum of differences
          plt.plot(range(0,len(probabilities)), sum_of_differences)
          plt.title('MoRF curve: Cumulative differences')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Sum of differences')
          plt.show()

      L = len(self.heatmap_regions)
      score = (1/(L+1)) * sum(differences)

      return round(score, 3)
  

def AOPC_Dataset(tensors, labels, model, heatmaps_regions, noise):
    
    '''
    Arguments
    tensors: will be torch.stack for dim=0 (thus 5dims in total; dim=0 from stacking and dims 1-4 from tensors)
    labels: we assume that labels are '0' to 'sth' (to match with argmax logic over the predicted prob distribution vector)
    heatmaps_regions: list of heatmap regions (i-th element corresponds to regions of the i-th tensor across tensors dim=0)
    noise: same as per MoRF class
    '''
    
    # to save the correctly classified image aopc scores 
    scores = []
      
    # A. check if correct prediction
      
    #iterates over first dimension and picks one tensor per iteration
    for i, tensor in enumerate(tensors):
      
      print(i)
      
      predicted_distribution = F.softmax(model(tensor), dim=1) # probability vector
      predicted_label = int(np.argmax(predicted_distribution.detach().numpy(), axis = 1))
      
      #B. for correctly predicted get aopc_score from class
      
      if labels[i] == predicted_label:
        score = MoRF(tensor, heatmaps_regions[i], model, noise).aopc()
        scores.append(score)
      
    # C. Average over all scores/tensors.
    #    Note that we should not multiply by (1/(L+1)) again. Already done in MoRF class.
    dataset_aopc_score = (1/len(scores)) * sum(scores)
      
    return round(dataset_aopc_score, 3)
