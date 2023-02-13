**Background**

As explained in [1] and [2], for CNNs ending in GAP layer followed by a fully connected layer (the "CAM architecture," e.g. ResNet, DenseNet etc), 
the visualizations produced by CAM [3], Grad-CAM [1], and HiResCAM [2] XAI algorithms are identical, as Grad-CAM and HiResCAM are alternative generalizations of CAM. 
In addition, one may find in [3] that the CAM explanations are **guaranteed** to reflect the locations the model used for calculating the class score.

<p align="center">
     <img src="https://user-images.githubusercontent.com/55101427/218502267-04f955ad-583f-471d-b9fe-8a6176f9918f.png" height="250" width="550" />
   </p>

However, for CNNs ending in one fully connected layer but with no preceding GAP layer the HiResCAM and Grad-CAM explanations are **no longer** identical.
As explained in [2], when the gradients are computed with respect to last convolutional layer, then HiResCAM explanation **provably** reflects the model's computations 
while the Grad-CAM explanation does not.

<p align="center">
     <img src="https://user-images.githubusercontent.com/55101427/218503517-dbc6f754-d487-4382-a5b4-ab48ef9a6552.png" height="300" width="550" />
   </p>

**Thesis summary**

The main focus of this thesis is to present **a quantitative comparison between GradCAM and HiResCAM explainations** in the context of "non-GAP" CNNs of Figure 3,
via attribution map evaluation metrics such as AOPC[4], Max-Sensitivity[5] and HAAS[6]. The experiments are conducted on medical image data and with custom written
variations of ReNet and VGG models.

**Sources**
  - [1] GradCAM: https://arxiv.org/pdf/1610.02391.pdf
  - [2] HiResCAM: https://arxiv.org/pdf/2011.08891.pdf
  - [3] CAM: https://arxiv.org/pdf/1512.04150.pdf
  - [4] AOPC: https://arxiv.org/pdf/1509.06321.pdf
  - [5] Max Sensitivity: https://arxiv.org/pdf/1901.09392.pdf
  - [6] HAAS: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759
