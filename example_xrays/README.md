As per HiResCAM official paper, if a CNN has structure "CNN part ending in Conv layer - Flatten - Raw class scores" and the gradients are computed with respect to the last convolutional layer, then the HiResCAM explanation provably reflects the model's computations while the Grad-CAM explanation does not.

Here, this theoretical result is quantified via the AOPC and Max Sensitivity scores. The results show that indeed the HiResCAM metric scores are better than the GradCAM ones. The experiment is conducted with a ResNet34 model customized as per the aforementioned structure.

Sources:
- HiResCAM: https://arxiv.org/pdf/2011.08891.pdf
- GradCAM: https://arxiv.org/pdf/1610.02391.pdf
- Dataset: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
