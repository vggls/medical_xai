<!-- *This repo contains material and code of the M.Sc. thesis in AI @ NCSR Demokritos & University of Piraeus <br>
originally titled "**Grad-CAM vs HiResCAM: A comparative study via quantitative evaluation metrics**" [text link](https://dione.lib.unipi.gr/xmlui/bitstream/handle/unipi/15495/Lamprou_mtn2107.pdf?sequence=1)* -->

## Title: On the evaluation of deep learning interpretability methods for medical images under the scope of faithfulness

## 1. Intro

We study the relationship between faithfulness and quantitative interpretability evaluation metrics in the context of attribution maps. 
Taking inspiration from the fact that a human would show preference to faithful attribution maps, it is natural to investigate whether attribution map metrics will assign them with better evaluation scores as well.

Our implementation fine-tunes pre-trained ResNet and VGG19 architectures over X-Rays ([Covid-19 Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database))
and digital pathology ([CRC](https://zenodo.org/record/1214456), 
[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000),
[BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis)) image datasets and calculates Grad-CAM[2] and HiResCAM[3] attribution maps, in a setting where only HiResCAM is known to produce faithful attributions (see [3]).
The objective is to evaluate the quality of the attribution maps via quantitative evaluation metrics such as AOPC[4], Max Sensitivity[5] and HAAS[6] and investigate if the metrics can distinguish between faithful and non-faithful attribution maps.

Our findings suggest that Max-Sensitivity and AOPC favour the HiResCAM attribution maps. This is attributed to the fact that this kind 
of mappings preserve the gradient effect on a pixel-level ultimately procducing high-resolution, informative and resilient maps.
On the other hand, the HAAS score does not contribute to our comparison as it evaluates almost all attribution maps as inaccurate. 
To this purpose we further compare our calculated values against values
obtained over a diverse group of models which are trained on non-medical benchmark datasets,
to eventually achieve more responsive results. The final experiments suggest that the metric is more
sensitive over complex medical patterns, commonly characterized by strong colour dependency
and multiple attention areas.

## 2. Background

Adopting the terminology of [3], an attribution map method will be considered ***faithful***
to a model if the sum of the attribution map values reflects the class score calculation, up to a class-dependent bias term deviation.

Based on results included in [1], [2] and [3], we summarize in the following table the relationship between Grad-CAM and HiResCAM in terms of values and faithfulness.
<p align="center">
     <img src="https://github.com/vggls/msc_thesis_medical_xai/assets/55101427/3db620a3-032b-43d8-a155-57dda47047c0.png" height="140" width="500" />
   </p>

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/deaf04b9-6818-4f75-b103-42cdc271f8fe.png" height="280" width="500" />
   </p>

## 3. Methodology
The combination of non-equivalent values and the faithfulness of HiResCAM turns the setup of *Conv - Flatten - Class Scores* structures, 
with gradients calculated at the last convolutional layer, into a *reference setting* where the algorithms are distinguishable. 
In the remaining setups, both algorithms are either equivalent or non-faithful. Consequently, this particular setup serves as a 
compelling ground for hosting a meaningful comparison between Grad-CAM and HiResCAM.

As a result, the workflow of our study is summarized as follows:

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/99c38eef-6713-4640-a5c1-160c3369f2f8.png" height="260" width="630" />
   </p>

## 4. Experimental results 
- AOPC score

In our experiments, we produce 224\*224 pixel Grad-CAM and HiResCAM attribution maps, which are perturbed by regions of size 56\*56, 28\*28, 21\*21 and 16\*16, resulting in heatmaps of size 4\*4, 8\*8, 11\*11 and 14\*14 respectively. In addition, per perturbation step, we replace the image pixels with re-sampled uniform noise. We note that large AOPC values suggest heatmaps of better quality.

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/357cba82-0bf2-49ee-bc92-d122740b2696.png" height="200" width="520" />
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
     <img src="https://github.com/vggls/medical_xai/assets/55101427/abdea6b2-0bc7-45f2-b7aa-4e55232133a1.png" height="90" width="470" />
   </p>

## 5. A short analysis on the results (Discussion)
- AOPC and Max Sensitivity
  
We observe that AOPC favors HiResCAM over Grad-CAM in 7/8 model experiments and Max Sensitivity in 8/8 model experiments. We discuss their results together since the analysis is rooted in the same reasoning.

A notable distinction between HiResCAM and Grad-CAM lies in the treatment of gradients. 
Grad-CAM calculates Gradient Averages to assign weights to the feature maps. On the other hand, HiResCAM uses the Hadamard product to weight each feature map pixel with its corresponding gradient, preserving in that way the gradient's value-and-sign influence on a pixel level.
Consequently, as explained in [3], HiResCAM generates fine-grained high resolution attribution maps, while Grad-CAM produces maps characterized by larger and smoother areas of interest because of the Gradient Averaging effect. This becomes evident in the following examples.

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/d6beb0ee-2213-40d4-b16c-e9605d056599.png" height="650" width="800" />
   </p>

Thus, the HiResCAM attribution maps provide more precise localization of the most discriminative regions, leading eventually to higher AOPC scores. Similarly, considering the Max-Sensitivity results, the high resolution HiResCAM maps develop a resilient behaviour to small perturbations in the input image.

- HAAS
  
In 15/16 cases we calculated a HAAS score below 1, implying that almost all attribution maps cannot capture the models' viewpoints.

Are these results related to the nature of medical images? 

In [6], HAAS was tested on datasets whose classes are determined by single objects and are not sensitive to subtle colour variations (Cifar10, STL10, ImageNet). On the other hand, medical images are more complex; the classes could have many attention areas and a stronger colour dependency. 
In the context of the positive Grad-CAM and HiResCAM attributions, we are interested in investigating if emphasizing the image pixels' intensity could potentially prevent the model from locating a learned pattern. 

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/6944d0a4-7536-449a-9c77-37a23068b078.png" height="150" width="500" />
     <img src="https://github.com/vggls/medical_xai/assets/55101427/9f72dc66-fe47-4473-8382-f6746130fa9e.png" height="150" width="500" />
   </p>
<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/9c41fd93-a5c0-443d-8c61-10c138070fb0.png" height="150" width="500" />
     <img src="https://github.com/vggls/medical_xai/assets/55101427/4db4bbf0-149d-477e-9067-22378c4abb9f.png" height="150" width="500" />
   </p>

We conduct the following experiment to further explore the relationship between HAAS and the medical data of this study: 
For Cifar10, STL10 and Imagenette we train a loop of 16 VGG19 models, configured over a variety of training batch size, learning rate, scheduler and weight decay (see in the text for more details), 
in order to track the HiResCAM HAAS scores range. 
The maximum and minimum HAAS values are reported in the following table, accompanied with the respective mean AUC score of the model.

<p align="center">
     <img src="https://github.com/vggls/medical_xai/assets/55101427/e2e30f4c-4580-4e06-936f-58e8c0821e20.png" height="120" width="350" />
   </p>

Per dataset, the pool of models yields maximum HAAS score above 1 for a highly performing model and minimum HAAS score slightly below 1 for a well performing model.
Hence, it was possible to derive meaningful HAAS scores when utilizing the non-medical datasets and models that were not optimally trained.
This stands in contrast with the medical data experimets of section 4 which were built on models that underwent meticulous training 
and suggests evidence that HAAS might be more sensitive to medical data.

## Main References

<sup>
     
- [1] [CAM](https://arxiv.org/pdf/1512.04150.pdf)
     
- [2] [GradCAM](https://arxiv.org/pdf/1610.02391.pdf)
  
- [3] [HiResCAM](https://arxiv.org/pdf/2011.08891.pdf)
  
- [4] [AOPC](https://arxiv.org/pdf/1509.06321.pdf)
  
- [5] [Max Sensitivity](https://arxiv.org/pdf/1901.09392.pdf)
  
- [6] [HAAS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9800759)
</sup>
