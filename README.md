**Background & Motivation**

We consider the gradient-based GradCAM[1] and HiResCAM[2] XAI algorithms for image data which were originally introduced as extentions of the well-known CAM[3] XAI algorithm. In general, they are both applicable to any CNN architecture with differentiable classifier part and allow to extract image explanations with respect to any convolutional block of the network. 

For the purposes of this study we restrict to cases where the classifier is a single full connected layer (the raw class scores layer) as in such structures, based on theory, HiResCAM explanations faithfully reflect the model's decisions. 

More specifically, in [1] and [2] one may find proof that for CNNs ending in GAP layer followed by a fully connected layer (the "CAM architecture" of Figure 1, e.g. ResNet, DenseNet etc), the visualizations produced by CAM, Grad-CAM, and HiResCAM XAI algorithms are identical. At the same time, in this architecture, as proved in [3], the CAM explanations are **guaranteed** to reflect the locations the model used for calculating the class score.

<p align="center">
     <img src="https://user-images.githubusercontent.com/55101427/218502267-04f955ad-583f-471d-b9fe-8a6176f9918f.png" height="250" width="550" />
   </p>

On the other hand, when the GAP layer is replaced by a Flatten layer (as per Figure 3) then one may find in [2] that the HiResCAM and Grad-CAM explanations are **no longer identical**. In this context, when calculating the gradients with respect to last convolutional layer then the HiResCAM explanation **provably** reflects the model's computations as it contributes directly to the raw class score calculation. At the same GradCAM fails to exhibit analogous behaviour. As explained in [2], GradCAM's gradient averaging step prevent's a direct connection between the raw class scores while visually this results in often expanding the attention area to  bigger and smoother areas which however are not faithful to the model's decisions.

<p align="center">
     <img src="https://user-images.githubusercontent.com/55101427/218503517-dbc6f754-d487-4382-a5b4-ab48ef9a6552.png" height="300" width="550" />
   </p>

**Thesis topic**

The main focus of this thesis is to present **a quantitative comparison between GradCAM and HiResCAM explainations** in the context of "non-GAP" CNNs of Figure 3 and see if the theoretical advantage of HiResCAM aligns well with results of attribution map evaluation metrics such as AOPC[4], Max-Sensitivity[5] and HAAS[6].

The experiments are conducted on medical image data with custom written variations of ReNet and VGG models.

**Main Sources**
  - [1] GradCAM: https://arxiv.org/pdf/1610.02391.pdf
  - [2] HiResCAM: https://arxiv.org/pdf/2011.08891.pdf
  - [3] CAM: https://arxiv.org/pdf/1512.04150.pdf
  - [4] AOPC: https://arxiv.org/pdf/1509.06321.pdf
  - [5] Max Sensitivity: https://arxiv.org/pdf/1901.09392.pdf
  - [6] HAAS: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759
