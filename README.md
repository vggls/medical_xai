## Title
**Grad-CAM vs HiResCAM: A comparative study via quantitative evaluation metrics**

## Abstract
In this study we utilize the Grad-CAM[2] and HiResCAM[3] attribution map methods and
consider a setting where the HiResCAM algorithm provably produces faithful explanations
while Grad-CAM does not. This theoretical result motivates us to investigate the
quality of their attribution maps in terms of quantitative evaluation metrics and examine
if faithfulness aligns with the metrics results. Our evaluation scheme implements
the well-established AOPC[4] and Max Sensitivity[5] scores along with the recently introduced
HAAS[6] score and utilizes ResNet and VGG pre-trained architectures trained on
the [CRC](https://zenodo.org/record/1214456), 
[Covid-19 Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), 
[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) 
and [BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis) medical image datasets. 
The experimental results suggest that Max-Sensitivity and
AOPC favour faithfulness. On the other hand, HAAS does not contribute meaningful
values to our comparison, but rather inspires further study about its nature.

Source: [Thesis link](https://dione.lib.unipi.gr/xmlui/bitstream/handle/unipi/15495/Lamprou_mtn2107.pdf?sequence=1)

## Background & Motivation (short version)

Adopting the terminology of [3], an attribution map method will be considered ***faithful***
to a model if the sum of the attribution map values reflects the class score calculation, up to a class-dependent bias term deviation.

Based on results included in [1], [2] and [3], the following table summarizes the relationship between Grad-CAM and HiResCAM in terms of values and faithfulness.
<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/3da9fb76-8da0-42f3-abe6-e36fa80792ca.png" height="140" width="500" />
   </p>

## Methodology
The combination of non-equivalent values and the faithfulness of HiResCAM turns the setup of *Conv - Flatten - Class Scores* structures, 
with gradients calculated at the last convolutional layer, into a *unique setup reference* where the algorithms can be distinguished. 
In the remaining setups, both algorithms are either equivalent or non-faithful. Consequently, this particular setup serves as a 
compelling ground for hosting a meaningful comparison between Grad-CAM and HiResCAM.

As a result, the workflow of our study is summarized as follows:

<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/5df15629-7fb4-4089-a6e0-3dce4cf3a83c.png" height="200" width="500" />
   </p>

## Experimental results 
***- AOPC***

In our experiments, we produce 224\*224 pixel Grad-CAM and HiResCAM attribution maps, which are perturbed by regions of size 56\*56, 28\*28, 21\*21 and 16\*16, resulting in heatmaps of size 4\*4, 8\*8, 11\*11 and 14\*14 respectively. In addition, per perturbation step, we replace the image pixels with re-sampled uniform noise. We note that large AOPC values suggest heatmaps of better quality.

<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/b963cba9-6d86-488e-ab17-0333035fdb73.png" height="200" width="500" />
   </p>

***- Max Sensitivity***

We calculated the metric for different levels r of increasing uniform noise perturbations and varied the number of perturbed instances y as follows: for r=0.05 and r=0.1 we drew y=20 samples, for r=0.2 and r=0.3 we drew y=30 samples and for r=0.4 and r=0.5 we drew y=40 samples. We note that low Max-Sensitivity values suggest heatmaps of better quality.

<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/137c00ab-40bc-468f-9afc-d0d9bfa749cc.png" height="450" width="450" />
   </p>

***- HAAS***

The HAAS score is free of hyper-parameters. We note that when HAAS is greater than 1, the attribution maps explain the features' importance well. On the other hand, if HAAS is less than 1, the attribution maps fail to bring out the features' importance for the model.

<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/fd22ced2-4f7d-4566-b2e5-d1fc7ebb787a.png" height="80" width="400" />
   </p>

An analysis of the results is provided in chapter 5 ("Discussion") of the thesis link provided in the Abstract.

## Main Sources
  - [1] [CAM](https://arxiv.org/pdf/1512.04150.pdf)
  - [2] [GradCAM](https://arxiv.org/pdf/1610.02391.pdf)
  - [3] [HiResCAM](https://arxiv.org/pdf/2011.08891.pdf)
  - [4] [AOPC](https://arxiv.org/pdf/1509.06321.pdf)
  - [5] [Max Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)
  - [6] [HAAS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759)
