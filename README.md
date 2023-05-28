## Title
**Grad-CAM vs HiResCAM: A comparative study via quantitative evaluation metrics**

## Abstract
In this study we utilize the Grad-CAM [1] and HiResCAM [2] attribution map methods and
consider a setting where the HiResCAM algorithm provably produces faithful explanations
while Grad-CAM does not. This theoretical result motivates us to investigate the
quality of their attribution maps in terms of quantitative evaluation metrics and examine
if faithfulness aligns with the metrics results. Our evaluation scheme implements
the well-established AOPC [3] and Max-Sensitivity [4] scores along with the recently introduced
HAAS [5] score and utilizes ResNet and VGG pre-trained architectures trained on
four medical image datasets.

## Background & Motivation (short version)

***For the purposes of this study***, an attribution map method will be considered ***faithful***
to a model if the sum of the attribution map values reflect the class score calculation.
Based on theory included in [2], when the CNN architecture is of the form ***Conv - Flatten - Class Scores*** and the XAI algorithm class
gradients are computed with respect to the last convolutional layer of the network, then
one can prove that HiResCAM is faithful to the model ([2] section 3.5.1).
On the other hand, Grad-CAM’s attribution maps do not exhibit analogous behaviour.
This fact means that HiResCAM attribution maps faithfully highlight the locations the
model identifies the class.

<p align="center">
     <img src="https://user-images.githubusercontent.com/55101427/218503517-dbc6f754-d487-4382-a5b4-ab48ef9a6552.png" height="300" width="550" />
   </p>

## Thesis topic

Motivated from this theoretical result, we want to quantify the quality of the Grad-CAM
and HiResCAM attribution maps in the above setting and examine if the AOPC, Max-
Sensitivity and HAAS metrics can distinguish between the faithful and non-faithful attribution maps.

## Main Sources
  - [1] [GradCAM](https://arxiv.org/pdf/1610.02391.pdf)
  - [2] [HiResCAM](https://arxiv.org/pdf/2011.08891.pdf)
  - [3] [AOPC](https://arxiv.org/pdf/1509.06321.pdf)
  - [4] [Max Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)
  - [5] [HAAS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759)
