## Title
**Grad-CAM vs HiResCAM: A comparative study via quantitative evaluation metrics**

## Abstract
In this study we utilize the Grad-CAM [2] and HiResCAM [3] attribution map methods and
consider a setting where the HiResCAM algorithm provably produces faithful explanations
while Grad-CAM does not. This theoretical result motivates us to investigate the
quality of their attribution maps in terms of quantitative evaluation metrics and examine
if faithfulness aligns with the metrics results. Our evaluation scheme implements
the well-established AOPC [4] and Max-Sensitivity [5] scores along with the recently introduced
HAAS [6] score and utilizes ResNet and VGG pre-trained architectures trained on
four medical image datasets. The experimental results suggest that Max-Sensitivity and
AOPC favour faithfulness. On the other hand, HAAS does not contribute meaningful
values to our comparison, but rather inspires further study about its nature.

Source: [Thesis link](https://dione.lib.unipi.gr/xmlui/bitstream/handle/unipi/15495/Lamprou_mtn2107.pdf?sequence=1)

## Background & Motivation (short version)

Adopting the terminology of [2], an attribution map method will be considered ***faithful***
to a model if the sum of the attribution map values reflect the class score calculation, up to a class-dependent bias term deviation.

Based on results included in [1], [2] and [3], the following table summarizes the relationship between Grad-CAM and HiResCAM in terms of values and faithfulness.
<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/bf7d6db1-571e-4308-987a-3e3c59112b6b.png" height="160" width="500" />
   </p>

The combination of non-equivalent values and the faithfulness of HiResCAM turns the setup of *Conv - Flatten - Class Scores* structures (as per Figure 3 below), 
with gradients calculated at the last convolutional layer, into a *unique setup reference* where the algorithms can be distinguished. 
In the remaining setups, both algorithms are either equivalent or non-faithful. Consequently, this particular setup serves as a 
compelling ground for hosting a meaningful comparison between Grad-CAM and HiResCAM.

<p align="center">
     <img src="https://user-images.githubusercontent.com/55101427/218503517-dbc6f754-d487-4382-a5b4-ab48ef9a6552.png" height="270" width="500" />
   </p>

## Workflow
<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/5df15629-7fb4-4089-a6e0-3dce4cf3a83c.png" height="250" width="600" />
   </p>

## Experimental results 
***AOPC***
<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/b963cba9-6d86-488e-ab17-0333035fdb73.png" height="220" width="500" />
   </p>

***Max Sensitivity***
<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/137c00ab-40bc-468f-9afc-d0d9bfa749cc.png" height="550" width="500" />
   </p>

***HAAS***
<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/fd22ced2-4f7d-4566-b2e5-d1fc7ebb787a.png" height="90" width="400" />
   </p>

An analysis of the results is provided in chapter 5 of the accompanying thesis link.

## Main Sources
  - [1] [CAM](https://arxiv.org/pdf/1512.04150.pdf)
  - [2] [GradCAM](https://arxiv.org/pdf/1610.02391.pdf)
  - [3] [HiResCAM](https://arxiv.org/pdf/2011.08891.pdf)
  - [4] [AOPC](https://arxiv.org/pdf/1509.06321.pdf)
  - [5] [Max Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)
  - [6] [HAAS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759)
