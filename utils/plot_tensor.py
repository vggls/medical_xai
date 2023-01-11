"""
Code that converts [-1,1]-values tensor to [0,1]-values tensor.

[-1,1]-value tensors arise from transforms of the form:
   'transforms.Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5])'

"""

def convert_for_plot(tensor):
    
    #convertion to [0 ,2] values
    tensor_image = tensor.cpu().permute(1,2,0) +1
    
    #convertion to [0 ,1] values
    tensor_image = tensor_image - tensor_image.min()
    tensor_image_0_1 = tensor_image / (tensor_image.max() - tensor_image.min())
    
    return tensor_image_0_1