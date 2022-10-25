
"""
Sources/Useful info

    => HiResCAM
        a) GitHub: https://github.com/jacobgil/pytorch-grad-cam
        b) Remarks on the target_layer attribute:
            - It is the layer wrt which we compute the (high prob) class derivatives
            - Usually this will be the last convolutional layer in the model.
              In this case, as remarked in Github by the authors, some common choices can be:
              Resnet18 and 50: model.layer4
              VGG, densenet161: model.features[-1]
              mnasnet1_0: model.layers[-1]
          
        c) Remark on the 'targets' parameter of the HiResCAM methdod:
             As per lines 105-110 from this code https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py
             as long as we are ok with getting only the heatmap of the highest prob category 
             we are ok with the below implementation and we do not have to set any value to this parameter (default is None)

    => DeepLIFT 
        a) Paper: https://arxiv.org/pdf/1704.02685.pdf (Includes links to video tutorial, ICML presentation and GitHub code)
        b) Captum DeepLift API:  https://captum.ai/api/deep_lift.html

    => LIME
        a) Paper: https://arxiv.org/pdf/1602.04938.pdf
        b) GitHub: https://github.com/marcotcr/lime
        c) 'lime_image' module: https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image
    
"""
# pip install grad-cam
# pip install captun
# pip install lime

import numpy as np
import functools
from skimage.measure import block_reduce

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision import transforms

from pytorch_grad_cam import HiResCAM
from captum.attr import DeepLift
from lime import lime_image

#-----------------------------------------------------------------------------------------------

def heatmap_function(tensor, model, xai_algorithm, target_layer=None):
    
    # no_of_regions = no. of regions per heatmap side (i.e. if heatmap is divided into 81 regions then no_of_regions=9)
    
    assert Tensor.dim(tensor) == 3
    
    assert xai_algorithm in ['hirescam', 'deeplift', 'lime']
    
    if xai_algorithm == 'hirescam':
        assert target_layer is not None
        positive_attributions = hirescam(tensor, model, target_layer)
    elif xai_algorithm == 'deeplift':
        _, positive_attributions = deeplift(tensor, model)
    elif xai_algorithm == 'lime':
        _, positive_attributions = lime(tensor, model)
        
    region_size = tensor.shape[1]//8
    
    heatmap = block_reduce(positive_attributions, (region_size, region_size), np.mean)  # AvgPooling
    
    regions_dict = {}
    for i in range(heatmap.shape[0]):
        for j in range(0, heatmap.shape[0]):
            regions_dict[i,j] = heatmap[i,j]
    regions = sorted(regions_dict.items(), key=lambda x: x[1], reverse=True)
    
    return heatmap, regions

#-----------------------------------------------------------------------------------------------

def hirescam(tensor, model, target_layer):

    cam = HiResCAM(model=model, target_layers=target_layer, use_cuda=False)

    attributions = cam(input_tensor=tensor.unsqueeze(0))[0,:,:]  # numpy.ndarray, 2-dim, values in [0,1]
    # note that attributions = positive_attributions
    
    return attributions

#-----------------------------------------------------------------------------------------------

def deeplift(tensor, model):
    
    tensor.requires_grad = True
    
    # If architecture has ReLU then set inplace to False (ex. vgg, densenet, mobilenet or hand-written)
    # If no ReLU then do nothing (ex. inception, efficientnets, googlenet)
    # Fails with resnet and inception; regardless of the check_ReLU_inplace method
    #       More info for resnet here --> https://github.com/pytorch/captum/issues/378
    if ('Inception' not in model.__class__.__name__) and \
       ('EfficientNet' not in model.__class__.__name__) and \
       ('GoogLeNet' not in model.__class__.__name__):
        print('ReLU path')
        model = ReLU_inplace_to_False(model)
    else:
        print('No ReLU path')
        pass

    attr_dl = DeepLift(model).attribute(tensor.unsqueeze(0),
                                        target=int(np.argmax(model(tensor.unsqueeze(0)).detach().numpy())))
    attributions3d = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0)) # numpy.ndarray, 3-dim, values pos+neg
    attributions = np.mean(attributions3d, axis = -1)                                   # numpy.ndarray, 2-dim, values pos+neg
    
    positive_attributions = np.where(attributions>0, attributions, 0)  # ReLU
    
    return attributions, positive_attributions

def ReLU_inplace_to_False(model):
    
    for layer in model._modules.values():
        
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)
        
    return model

#-----------------------------------------------------------------------------------------------

def batch_predict(model, images):
    
    model.eval()
    batch = torch.stack(tuple(torch.from_numpy(i).permute(2,1,0) for i in images), dim=0) # is this correct ?
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch.float())
    probs = F.softmax(logits, dim=1)
    
    return probs.detach().cpu().numpy()

def lime(tensor, model):
    
    #tensor.requires_grad = False
    
    transform = transforms.ToPILImage()
    img = transform(tensor)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(img),           
                                         classifier_fn=functools.partial(batch_predict, model),
                                         #classifier_fn=batch_predict,
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=1000)  # number of images that will be sent to classification function

    exp_class = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[exp_class])
    attributions = np.vectorize(dict_heatmap.get)(explanation.segments)     # numpy.ndarray, 2-dim, values pos+neg

    positive_attributions = np.where(attributions>0, attributions, 0)
    
    return attributions, positive_attributions

#-----------------------------------------------------------------------------------------------