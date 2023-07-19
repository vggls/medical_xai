### Title
Grad-CAM vs HiResCAM: A comparative study via quantitative evaluation metrics

### 1. Intro
This repository contains the source code of the study conducted in 
[this text link](https://dione.lib.unipi.gr/xmlui/bitstream/handle/unipi/15495/Lamprou_mtn2107.pdf?sequence=1).

We address the problem of quantifying the quality of attribution maps in a setting where HiResCAM produces *faithful* attributions while Grad-CAM does not.
Our evaluation scheme implements
the well-established AOPC[4] and Max Sensitivity[5] scores along with the recently introduced
HAAS[6] score and utilizes ResNet and VGG pre-trained architectures trained on
the [CRC](https://zenodo.org/record/1214456), 
[Covid-19 Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), 
[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) 
and [BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis) medical image datasets. 
Our findings (see below) suggest that Max-Sensitivity and AOPC results align with the faithful attribution maps.
On the other hand, the HAAS score does not contribute to our comparison as it evaluates almost all attribution maps as inaccurate. 
This inspires further study about the nature of HA images and led us to investigate their relation with class features 
which could potentially vary between medical and non-medical datasets.

### 2. Background

Adopting the terminology of [3], an attribution map method will be considered ***faithful***
to a model if the sum of the attribution map values reflects the class score calculation, up to a class-dependent bias term deviation.

Based on results included in [1], [2] and [3], we summarize in the following table the relationship between Grad-CAM and HiResCAM in terms of values and faithfulness.
<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/3db620a3-032b-43d8-a155-57dda47047c0.png" height="140" width="500" />
   </p>

### 3. Methodology
The combination of non-equivalent values and the faithfulness of HiResCAM turns the setup of *Conv - Flatten - Class Scores* structures, 
with gradients calculated at the last convolutional layer, into a *unique setup reference* where the algorithms can be distinguished. 
In the remaining setups, both algorithms are either equivalent or non-faithful. Consequently, this particular setup serves as a 
compelling ground for hosting a meaningful comparison between Grad-CAM and HiResCAM.

As a result, the workflow of our study is summarized as follows:

<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/23fec2ee-6178-47c5-bf8a-1c4d93800b9e.png" height="230" width="580" />
   </p>
   
### 4. Experimental results 
***- AOPC score***

In our experiments, we produce 224\*224 pixel Grad-CAM and HiResCAM attribution maps, which are perturbed by regions of size 56\*56, 28\*28, 21\*21 and 16\*16, resulting in heatmaps of size 4\*4, 8\*8, 11\*11 and 14\*14 respectively. In addition, per perturbation step, we replace the image pixels with re-sampled uniform noise. We note that large AOPC values suggest heatmaps of better quality.

<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/9d97f82e-6e22-44c9-924d-600e992363b9.png" height="210" width="550" />
   </p>

***- Max Sensitivity score***

We calculated the metric for different levels r of increasing uniform noise perturbations and varied the number of perturbed instances y as follows: for r=0.05 and r=0.1 we drew y=20 samples, for r=0.2 and r=0.3 we drew y=30 samples and for r=0.4 and r=0.5 we drew y=40 samples. We note that low Max-Sensitivity values suggest heatmaps of better quality.

Blue line: Grad-CAM, Red line: HiResCAM
<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/00823cf5-f13f-4c30-9f25-f1eb95c0c012.png" height="570" width="650" />
   </p>

***- HAAS score***

The HAAS score is free of hyper-parameters. We note that when HAAS is greater than 1, the attribution maps explain the features' importance well. On the other hand, if HAAS is less than 1, the attribution maps fail to bring out the features' importance for the model.

<p align="left">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/9d78612f-8fab-41b6-8bb8-0e5e3e5fc842.png" height="90" width="470" />
   </p>

An analysis of the results is provided in section 5 ("Discussion") of the link provided in the "1. Intro".

#### Main References

- [1] [CAM](https://arxiv.org/pdf/1512.04150.pdf)
- [2] [GradCAM](https://arxiv.org/pdf/1610.02391.pdf)
- [3] [HiResCAM](https://arxiv.org/pdf/2011.08891.pdf)
- [4] [AOPC](https://arxiv.org/pdf/1509.06321.pdf)
- [5] [Max Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)
- [6] [HAAS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759)
