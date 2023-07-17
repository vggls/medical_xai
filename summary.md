### Summary

In this study we consider the Grad-CAM and HiResCAM eXplainable AI (a.k.a. XAI) algorithms 
which are gradient based techniques that produce class-specific attribution maps. 
The two methods differ in the way they utilize the gradient effect as HiResCAM preserves it
on a pixel level while Grad-CAM calculates averages over feature map pixels.

We adopt the terminology of the HiResCAM publication and
consider an attribution map method to be faithful
to a model if the sum of the attribution map values reflects the class score calculation, 
deviating up to a class-dependent bias term.
Based on theory, when the CNN architecture is of the form Conv layer - Flatten layer - Class Scores 
and the XAI algorithm gradients are computed with respect to the last convolutional layer of the network, 
then one can prove that HiResCAM is faithful to the model and the produced attribution maps accurately 
highlight the pixel locations the model identifies the class. 
On the other hand, Grad-CAM’s attribution maps do not exhibit analogous behaviour.

This theoretical result motivates us to investigate the
quality of their attribution maps in terms of quantitative evaluation metrics 
and examine if faithfulness aligns with the metrics results. Our evaluation 
scheme implements the well-established AOPC and Max-Sensitivity scores along 
with the recently introduced HAAS score and utilizes ResNet and VGG pre-trained 
architectures trained on four well-known medical image datasets; 
CRC, COVID-19 Radiography Database, HAM10000 and BreakHis.

The experimental results suggest that Max-Sensitivity and
AOPC favour the HiResCAM faithful attribution maps over the non-faithful Grad-CAM attribution maps. 
This is attributed to the fact that HiResCAM fully exploits the gradients in terms of values and sign, 
producing high resolution HiResCAM attribution maps which provide a more precise class localization. 
On the other hand, HAAS did not contribute meaningful values to our comparison, as in almost all experiments 
suggested that the attribution maps are inaccurate, regardless of dataset, model and attribution method. 
This inspires further study and led us to investigate its relation with characteristics such as
the class pattern distribution and the class colour.
