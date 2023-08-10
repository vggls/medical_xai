## Title: A comparative study of Grad-CAM vs HiResCAM interpretability methods under the scope of faithfulness

## 1. Intro
This repository contains the source code of the study presented in the following
[text link](https://dione.lib.unipi.gr/xmlui/bitstream/handle/unipi/15495/Lamprou_mtn2107.pdf?sequence=1).
<!--- to comment out --->

We address the problem of quantifying the quality of attribution maps in a setting where HiResCAM[3] produces *faithful* attributions while Grad-CAM[2] does not,
under the objective to investigate whether faithfulness aligns with the metrics results.
Our evaluation scheme implements the well-established AOPC[4] and Max Sensitivity[5] scores along with the recently introduced HAAS[6] score and utilizes ResNet 
and VGG pre-trained architectures trained on X-Rays ([Covid-19 Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database))
and digital pathology ([CRC](https://zenodo.org/record/1214456), 
[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000),
[BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis)) medical image datasets.

Our findings (see below) suggest that Max-Sensitivity and AOPC results favour the faithful attribution maps.
On the other hand, the HAAS score does not contribute to our comparison as it evaluates almost all attribution maps as inaccurate. 
This inspires further study about the nature of HA images and leads us to investigate their relation with features 
which could potentially vary between medical and non-medical datasets.

## 2. Background

Adopting the terminology of [3], an attribution map method will be considered ***faithful***
to a model if the sum of the attribution map values reflects the class score calculation, up to a class-dependent bias term deviation.

Based on results included in [1], [2] and [3], we summarize in the following table the relationship between Grad-CAM and HiResCAM in terms of values and faithfulness.
<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/3db620a3-032b-43d8-a155-57dda47047c0.png" height="140" width="500" />
   </p>

## 3. Methodology
The combination of non-equivalent values and the faithfulness of HiResCAM turns the setup of *Conv - Flatten - Class Scores* structures, 
with gradients calculated at the last convolutional layer, into a *unique setup reference* where the algorithms can be distinguished. 
In the remaining setups, both algorithms are either equivalent or non-faithful. Consequently, this particular setup serves as a 
compelling ground for hosting a meaningful comparison between Grad-CAM and HiResCAM.

As a result, the workflow of our study is summarized as follows:

<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/23fec2ee-6178-47c5-bf8a-1c4d93800b9e.png" height="230" width="580" />
   </p>
   
## 4. Experimental results 
- AOPC score

In our experiments, we produce 224\*224 pixel Grad-CAM and HiResCAM attribution maps, which are perturbed by regions of size 56\*56, 28\*28, 21\*21 and 16\*16, resulting in heatmaps of size 4\*4, 8\*8, 11\*11 and 14\*14 respectively. In addition, per perturbation step, we replace the image pixels with re-sampled uniform noise. We note that large AOPC values suggest heatmaps of better quality.

<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/9d97f82e-6e22-44c9-924d-600e992363b9.png" height="210" width="550" />
   </p>

- Max Sensitivity score

We calculated the metric for different levels r of increasing uniform noise perturbations and varied the number of perturbed instances y as follows: for r=0.05 and r=0.1 we drew y=20 samples, for r=0.2 and r=0.3 we drew y=30 samples and for r=0.4 and r=0.5 we drew y=40 samples. We note that low Max-Sensitivity values suggest heatmaps of better quality.

Blue line: Grad-CAM, Red line: HiResCAM
<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/00823cf5-f13f-4c30-9f25-f1eb95c0c012.png" height="570" width="650" />
   </p>

- HAAS score

The HAAS score is free of hyper-parameters. We note that when HAAS is greater than 1, the attribution maps explain the features' importance well. On the other hand, if HAAS is less than 1, the attribution maps fail to bring out the features' importance for the model.

<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/9d78612f-8fab-41b6-8bb8-0e5e3e5fc842.png" height="90" width="470" />
   </p>

## 5. A short analysis on the results (Discussion)
- AOPC and Max Sensitivity
  
We observe that AOPC favors HiResCAM over Grad-CAM in 7/8 model experiments and Max Sensitivity in 8/8 model experiments. We discuss their results together since the analysis is rooted in the same reasoning.

A notable distinction between HiResCAM and Grad-CAM lies in the treatment of gradients. 
Grad-CAM calculates Gradient Averages to assign weights to the feature maps. On the other hand, HiResCAM uses the Hadamard product to weight each feature map pixel with its corresponding gradient, preserving in that way the gradient's value-and-sign influence on a pixel level.
Consequently, as explained in [3], HiResCAM generates fine-grained high resolution attribution maps, while Grad-CAM produces maps characterized by larger and smoother areas of interest because of the Gradient Averaging effect. This becomes evident in the following examples.

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/5f520ba6-369a-491a-82cc-572210a7c511.png" height="300" width="390" />
     <img src="https://github.com/vggls/medical_xai/assets/55101427/06622816-4bee-4c4b-921b-3aa2dab1d272.png" height="300" width="390" />
   </p>

Thus, the HiResCAM attribution maps provide more precise localization of the most discriminative regions, leading eventually to higher AOPC scores. Similarly, considering the Max-Sensitivity results, the high resolution HiResCAM maps develop a resilient behaviour to small perturbations in the input image.

- HAAS
  
In 15/16 cases we calculated a HAAS score below 1, implying that almost all attribution maps cannot capture the models' viewpoints.

Are these results related to the nature of medical images? 

HAAS in [6] was tested on datasets whose classes are determined by single objects and are not sensitive to subtle colour variations (Cifar10, STL10, ImageNet). On the other hand, medical images are more complex; the classes could have many attention areas and a stronger colour dependency. As a result,
in the context of the positive Grad-CAM and HiResCAM attributions, we are interested in investigating if emphasizing the image pixels intensity could potentially prevent the model from 
locating a learned pattern. Some examples of this scenario are included in the following image.

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/d744776e-a534-468a-8df9-0f2583229b1b.png" height="120" width="500" />
     <img src="https://github.com/vggls/medical_xai/assets/55101427/b6187c1c-5928-4ce9-836b-87e7d1c86c46.png" height="120" width="500" />
   </p>
<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/b4552306-0605-4120-a474-bccb47074ff9.png" height="120" width="500" />
     <img src="https://github.com/vggls/medical_xai/assets/55101427/cf84ad84-6b11-4561-83eb-d32bca01027d.png" height="120" width="500" />
   </p>

We conduct the following experiment to further explore the relationship between HAAS and the medical data of this study: For Cifar10, STL10 and Imagenette we train a loop of 16 VGG19 models, configured over a variety of training batch size, learning rate, scheduler and weight decay (see in the text for more details), in order to track the HiResCAM HAAS scores range. 
The maximum and minimum HAAS values are included in the following table, accompanied with the respective mean AUC score of the model.

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/33190fdf-a4c6-4287-8918-b048c112213f.png" height="120" width="350" />
   </p>

Per dataset, the pool of models yields maximum HAAS score above 1 for a highly performing model and minimum HAAS score slightly below 1 for a well performing model.
Hence, it was possible to derive meaningful HAAS scores when utilizing the non-medical datasets and models that were not optimally trained.
This stands in contrast with the medical data experimets of section 4 which were built on models that underwent meticulous training 
and suggests evidence that HAAS might be more sensitive to medical data.

## Main References

- [1] [CAM](https://arxiv.org/pdf/1512.04150.pdf)
- [2] [GradCAM](https://arxiv.org/pdf/1610.02391.pdf)
- [3] [HiResCAM](https://arxiv.org/pdf/2011.08891.pdf)
- [4] [AOPC](https://arxiv.org/pdf/1509.06321.pdf)
- [5] [Max Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)
- [6] [HAAS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759)
