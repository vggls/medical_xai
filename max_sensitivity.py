'''
Sources
    - Official Paper: https://arxiv.org/pdf/1901.09392.pdf
    - Official GitHub: https://github.com/chihkuanyeh/saliency_evaluation (link included in page 2 of the paper)
    
Remark
    The methods of the 'Image level calculations' section are taken from the 'infid_sen_utils.py' file of the authors' 
    aforementioned GitHub page and adjusted accordingly to pytorch_grad_cam class instances.
    The methods of the 'Dataset level calculations' section are custom written functions that extend the notion of the
    Max Sensitivity metric to a Dataset level (data are loaded via Dataloaders). This is achieved by calculating the
    average Max Sensitivity value over the correctly classified images.
    
'''


import math
import numpy as np
import torch
from collections import OrderedDict
import plotly.graph_objects as go

'''
Method Arguments:
    X: is a 4-dim image tensor (1,C,W,H)
    cam_instance: pytorch_grad_cam instance
                  ex. cam_instance = HiResCAM(model=model, target_layers=[model.conv2], use_cuda=True)
    expl: pixel level attributions matrix (explanations) as extracted from get_explanation method. Shape is (1,1,W,H)
    radius: the radius of the X-centered ball
    iterations: no. of sampled points from the Ball(X,radius)

'''

# ------------------------ Image level calculations ----------------------------------------------

def sample_eps_Inf(image, epsilon):

    # function that creates uniform random sample in [-epsilon,epsilon] in the image size

    N=1
    images = np.tile(image, (N, 1, 1, 1))

    dim = images.shape

    return np.random.uniform(low = -1 * epsilon, high=epsilon, size=dim)

def get_explanation(X, model, cam_instance):

    pixel_attributions = cam_instance(input_tensor=X)

    expl = np.reshape(pixel_attributions, (1, pixel_attributions.shape[0], 
                                           pixel_attributions.shape[1], 
                                           pixel_attributions.shape[2]))

    return expl # (1,1,sth,sth)

def get_exp_sens(X, model, expl, cam_instance, radius, iterations):

    '''
    Remark: If norm(expl) very small then max sensitivity will be inf as per below computation.
            In the code we skip this kind of cases by returning None value.
    '''

    X = X.cuda()

    max_diff = -math.inf

    norm = np.linalg.norm(expl) # as per get_expl_infid_sens method of the vis_mnist.ipynb notebook

    if norm>1e-7:

        for _ in range(iterations):

            sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), radius)).cuda()

            X_noisy = X + sample #Here values might go beyond [-1,1]. 
                                 #Corresponds to 'y' of the paper. It is a point in the Ball(center=X, radius=radius)

            expl_eps = get_explanation(X_noisy, model, cam_instance)

            max_diff = max(max_diff, np.linalg.norm(expl-expl_eps)/norm)
        
        expl_max_sensitivity = max_diff

        return expl_max_sensitivity
    
    else:

        return None
    

#------------------------ Dataset level calculations -------------------------------------------------

def MaxSensitivity_Dataset(dataloader, model, cam_instance, radius, iterations):

    model = model.cuda()

    data_scores = []

    i = 0 #tracks number of all test images
    j = 0 #tracks number of correctly classified test images

    for batch_images, batch_labels in dataloader:

        for image, label in zip(batch_images, batch_labels):
            image, label = image.cuda(), label.cuda()
            
            real_label = int(label.cpu().detach().numpy())
            pred_label = int(torch.argmax(model(image.unsqueeze(0)), dim=1).cpu().detach().numpy())

            i += 1

            if real_label == pred_label:
                
                j += 1

                expl = get_explanation(image.unsqueeze(0), model, cam_instance)
                score = get_exp_sens(image.unsqueeze(0), model, expl, cam_instance, radius, iterations)
                if score != None:
                    data_scores.append(score)
  
    mean_score = sum(data_scores)/len(data_scores)

    print('Correctly predicted images: {}/{}'.format(j,i))
    print('No. of correctly classified images that were skipped due to almost 0 explanation matrix norm: {}'.format(j - len(data_scores)))
        
    return round(mean_score, 3), data_scores

def plot_scores_frequency(data_scores):
    
    keyDict = ('[0-1)', '[1-10)', '[10-100)', '[100-1000)', '[1000-~)')
    dictionary = OrderedDict([(key, []) for key in keyDict])

    dictionary['[0-1)'] = len([x for x in data_scores if (x>=0 and x<1)])
    dictionary['[1-10)'] = len([x for x in data_scores if (x>=1 and x<10)])
    dictionary['[10-100)'] = len([x for x in data_scores if (x>=10 and x<100)])
    dictionary['[100-1000)'] = len([x for x in data_scores if (x>=100 and x<1000)])
    dictionary['[1000-~)'] = len([x for x in data_scores if (x>=1000)])

    trace = go.Bar(x=list(dictionary.keys()), 
                    y=list(dictionary.values()),
                    text=list(dictionary.values()), 
                    textposition='outside')
    layout = go.Layout(title="Frequency of Max Sensitivity Values",
                        width=700, height=550, 
                        title_x=0.5,
                        bargap = 0.4
                        )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()