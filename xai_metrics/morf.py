"""
Sources
    - Paper : https://arxiv.org/pdf/1509.06321.pdf 
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

#-------Image Level---------------------------------------------------------------------------------------------------

class MoRF():

  def __init__(self, tensor, heatmap_regions, model):

      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
      self.tensor = tensor                    # tensor (1,c,h,w)
      assert(self.tensor.shape[2]==self.tensor.shape[3])
      
      self.heatmap_regions = heatmap_regions  # the image heatmap regions in order of importance (resulting from a XAI algorithm)
      self.model = model                      # a neural network model WITH Softmax in the end of the classifier
      
      #  the size of the image region that will be noise perturbed
      if self.tensor.shape[3]%np.sqrt(len(self.heatmap_regions)) == 0:
          self.perturbation_size = int(self.tensor.shape[3]/np.sqrt(len(self.heatmap_regions)))
      else:
          self.perturbation_size = int(self.tensor.shape[3]/np.sqrt(len(self.heatmap_regions)))+1
            
      isinstance(self.tensor, torch.Tensor)

      self.model.to(self.device)
      self.tensor.to(self.device)


  def perturbations(self, plot_morf_curve=False):

      # list to store all predicted probabilities of the top predicted class
      perturbations = []
      softmax = nn.Softmax(dim=1)

      # top predicted class info
      raw_scores = self.model(self.tensor)            # tensor of raw class scores
      probs = softmax(raw_scores)                     # probability distribution (via softmax)
      index = torch.argmax(probs)                     # index of the top predicted class
      class_prob = round(float(probs[0,index]),3)     # probability of the top predicted class
      perturbations.append(class_prob)

      image = self.tensor.detach().clone()            # do this in order to have access to the initial tensor after the perturbations

      noise_shape = (3, self.perturbation_size, self.perturbation_size)

      for step in range(0, len(self.heatmap_regions)):
        
        # update noise 
        uniform_noise = (torch.min(self.tensor) - torch.max(self.tensor)) * torch.rand(noise_shape).to(self.device) + torch.max(self.tensor)
        
        # get region row and column
        region_row = self.heatmap_regions[step][0][0]
        region_column = self.heatmap_regions[step][0][1]
        max_row_column = np.sqrt(len(self.heatmap_regions)) - 1

        # apply noise to tensor region - case1 noise fits image perfectly, case2 noise should be adjusted for the image edges
        if self.tensor.shape[3]%np.sqrt(len(self.heatmap_regions)) == 0:
            image[:,:3,region_row*self.perturbation_size:(region_row+1)*self.perturbation_size, \
                   region_column*self.perturbation_size:(region_column+1)*self.perturbation_size] = uniform_noise
        else:
            if (region_row < max_row_column) and (region_column < max_row_column):
                image[:,:3,region_row*self.perturbation_size:(region_row+1)*self.perturbation_size, \
                   region_column*self.perturbation_size:(region_column+1)*self.perturbation_size] = uniform_noise
            elif (region_row == max_row_column) and (region_column < max_row_column):
                remaining_rows = self.tensor.shape[3] - region_row*self.perturbation_size
                image[:,:3,-remaining_rows:, region_column*self.perturbation_size:(region_column+1)*self.perturbation_size] = \
                    uniform_noise[:3, :remaining_rows,:]
            elif (region_row < max_row_column) and (region_column == max_row_column):
                remaining_columns = self.tensor.shape[3] - region_column*self.perturbation_size
                image[:,:3,region_row*self.perturbation_size:(region_row+1)*self.perturbation_size, -remaining_columns: ] = \
                    uniform_noise[:3, :, :remaining_columns]
            elif(region_row == max_row_column) and (region_column == max_row_column):
                remaining_rows = self.tensor.shape[3] - region_row*self.perturbation_size
                remaining_columns = self.tensor.shape[3] - region_column*self.perturbation_size
                image[:,:3, -remaining_rows:, -remaining_columns: ] = uniform_noise[:3, :remaining_rows, :remaining_columns]
                
        #plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
        #plt.show()
        # make prediction and get value via the index used above (see prob variable line). this value will be the class predicted probability
        perturbed_raw_scores = self.model(image)
        perturbed_probs = softmax(perturbed_raw_scores)
        perturbed_prob = round(float(perturbed_probs[0,index]), 3)
        
        # append the value to perturbations
        perturbations.append(perturbed_prob)
    
      del image
      del uniform_noise
      gc.collect()
    
      if plot_morf_curve==True:
          # A good heatmap results in a large area over the morf (perturbation) curve. This is where the 'AOPC' name comes from.
          # Note that this area is controlled by the sum of differences f(x_0) - f(x_step). This is why we compute them in the 'aopc' method.
          plt.plot(range(0,len(perturbations)), perturbations)
          plt.title('MoRF Perturbation Curve')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Predicted probability')
          plt.show()

      return perturbations

  def aopc(self, plot_step_differences=False, plot_cumulative_differences=False):

      probabilities = self.perturbations()

      unpert_prob = [probabilities[0] for _ in range(len(probabilities))]
      differences = [a - b for a, b in zip(unpert_prob, probabilities)] #raw, un-normalized

      if plot_step_differences==True:
          plt.plot(range(0,len(probabilities)), differences)
          plt.title('Differences: f(x_0) - f(x_step)')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Probabilities difference')
          plt.show()

      if plot_cumulative_differences==True:
          # Note that the area below this curve is the AOPC score.
          sum_of_differences = np.cumsum(differences)   #cumulative sum of differences
          plt.plot(range(0,len(probabilities)), sum_of_differences)
          plt.title('Cumulative differences')
          plt.xlabel('Perturbation steps')
          plt.ylabel('Sum of differences')
          plt.show()
          
      # compute image aopc score
      L = len(self.heatmap_regions)
      score = (1/(L+1)) * sum(differences)

      return differences, round(score, 3)

#------Dataset Level----------------------------------------------------------------------------------------------------

def AOPC_Dataset(dataset, 
                 model, 
                 region_size,
                 cam_instance):
    
    '''
    Remarks: Instead of appending image aopc scores we could have alternatively update the total score (aopc_score+=img_score)
            and we could have also included the normalization steps in the end of the method.
            However, both these approaches would prevent us from aggregating multiple batch scores which is desirabale.
    '''
    
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
