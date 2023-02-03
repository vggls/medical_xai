**--> Summary**

As per HiResCAM official paper, if a CNN has structure "CNN part ending in Conv layer - Flatten - Raw class scores" and the gradients are computed with respect to the last convolutional layer, then the HiResCAM explanation provably reflects the model's computations while the Grad-CAM explanation does not.

Here, this theoretical result is quantified via the AOPC and Max Sensitivity scores. The results show that indeed the HiResCAM metric scores are better than the GradCAM ones. The experiment is conducted with a ResNet34 model customized as per the aforementioned structure.

**--> Sources**
- HiResCAM: https://arxiv.org/pdf/2011.08891.pdf
- GradCAM: https://arxiv.org/pdf/1610.02391.pdf
- Dataset: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

**--> ResNet34 Testing Results**

<p float="left">
     <img src="https://user-images.githubusercontent.com/55101427/216668907-4a064f8f-f928-429b-a90c-137aac450513.png" height="170" width="300" />
     <img src="https://user-images.githubusercontent.com/55101427/216669061-4cfceabe-6e69-4436-89d6-20f4add61671.png" height="250" width="350" />
   </p>

**--> Max Sensitivity Results**
<p float="left">
     <img src="https://user-images.githubusercontent.com/55101427/216670559-ed723513-0f6c-409a-a8ce-90b649feab6a.png" height="270" width="600" />
   </p>

**--> AOPC Results**
<p float="left">
     <img src="https://user-images.githubusercontent.com/55101427/216671018-082f2e0e-c689-4052-b0dd-922e6161aff8.png" height="180" width="300" />
   </p>
   
<p float="left">
     <img src="https://user-images.githubusercontent.com/55101427/216671949-ea81d1c7-db25-4e74-a4c9-392d58684e9d.png" height="250" width="500" />
     <img src="https://user-images.githubusercontent.com/55101427/216671503-40d2644b-bff2-4d67-aa71-b3a088b0af6d.png" height="250" width="500" />
   </p>
   
<p float="left">
   <img src="https://user-images.githubusercontent.com/55101427/216672311-efaaafcd-3d66-477c-abf2-60daac8adc71.png" height="250" width="500" />
   <img src="https://user-images.githubusercontent.com/55101427/216672591-34bcfc7f-71a9-4abb-ba16-d53710eeb6ee.png" height="250" width="500" />
 </p>
