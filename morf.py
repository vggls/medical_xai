"""
Sources
    - Paper : https://arxiv.org/pdf/1509.06321.pdf 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time


from heatmap import Heatmap

#-------Image Level---------------------------------------------------------------------------------------------------

class MoRF():

  # Rmk: tensor side should be resized to a multiplier of 8 (ex 64, 128, 256) to align with heatmaps.py region_size calculations

  def __init__(self, tensor, heatmap_regions, model):

      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
      self.tensor = tensor                    # tensor (1,c,h,w)
      assert(self.tensor.shape[2]==self.tensor.shape[3])
      
      self.heatmap_regions = heatmap_regions  # the image heatmap regions in order of importance (resulting from a XAI algorithm)
      self.model = model                      # a neural network model WITH Softmax in the end of the classifier
        
      self.perturbation_size = int(self.tensor.shape[3]/np.sqrt(len(self.heatmap_regions)))       #  the size of the image region that will be perturbed

      isinstance(self.tensor, torch.Tensor)

      self.model.to(self.device)
      self.tensor.to(self.device)


  def perturbations(self, plot_morf_curve=False):

      # list to store all predicted probabilities of the top predicted class
      perturbations = []

      # top predicted class info
      predict = self.model(self.tensor)
      index = torch.argmax(predict)                   # index of the top predicted class
      prob = round(float(predict[0,index]),3)         # probability of the top predicted class
      perturbations.append(prob)

      image = self.tensor.detach().clone()            # do this in order to have access to the initial tensor after the perturbations

      noise_shape = (3, self.perturbation_size, self.perturbation_size)

      for step in range(0, len(self.heatmap_regions)):
        
        # update noise 
        uniform_noise = (torch.min(self.tensor) - torch.max(self.tensor)) * torch.rand(noise_shape).to(self.device) + torch.max(self.tensor)
        
        # get region row and column
        region_row = self.heatmap_regions[step][0][0]
        region_column = self.heatmap_regions[step][0][1]

        # apply noise to tensor region
        image[:,:3,region_row*self.perturbation_size:(region_row+1)*self.perturbation_size, \
               region_column*self.perturbation_size:(region_column+1)*self.perturbation_size] = uniform_noise

        # make prediction and get value via the index used above (see prob variable line). this value will be the class predicted probability
        perturbed_prob = round(float(self.model(image)[0,index]), 3)
        
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

  def aopc(self, plot_step_differences=False, plot_cumulative_differences=False):

      probabilities = self.perturbations()

      unpert_prob = [probabilities[0] for _ in range(len(probabilities))]
      differences = [a - b for a, b in zip(unpert_prob, probabilities)]

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

      L = len(self.heatmap_regions)
      score = (1/(L+1)) * sum(differences)

      return differences, round(score, 2)

#------Dataset Level----------------------------------------------------------------------------------------------------

def AOPC_Dataset(dataset, 
                 model, 
                 region_size,
                 cam_instance):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    differences=[]; scores = []
        
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
            
            differences.append(img_diffs)  #for the plot
            scores.append(img_score)       #for total aopc score
      
    print('Total time: {} secs'.format(total_time))
    print('No of correctly classified images: {}/{}'.format(no_of_correctly_classified, len(dataset)))
    print('Avg secs per image: ', round(total_time/no_of_correctly_classified, 2))
       
    # Average over all image scores to calculate the final score
    # Note that we should not multiply by (1/(L+1)) again. Already done in MoRF class.
    dataset_aopc_score = (1/len(scores)) * sum(scores)
      
    return differences, scores, round(dataset_aopc_score, 2)
    

def normalize(lst, steps):
    return [x / (steps+1) for x in lst]

def plot_aopc_per_step(differences):
    
    # Construct the AOPC-vs-Perturbation step L plot (over the entire dataset)
    
    steps = len(differences[0])-1
    normalized_differences = [normalize(diff, steps) for diff in differences]

    differences_per_step = np.sum(normalized_differences, axis=0)
    cumulative = list(np.cumsum(differences_per_step))
    no_of_points = len(differences)
    cumulative = [x * (1/no_of_points) for x in cumulative]
    plt.plot(range(0,len(cumulative)), cumulative)
    plt.title('AOPC(L) -vs- Perturbation step L')
    plt.xlabel('Perturbation step L')
    plt.ylabel('AOPC(L)')
    plt.show()
    
    
