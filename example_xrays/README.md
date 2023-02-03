As per HiResCAM official paper, for CNNs with the structure "CNN part ending in Conv layer - Flatten - Raw class scores" the HiResCAM explanation provably reflects the model's computations while the Grad-CAM explanation does not.

Here, this theoretical result is quantified via the AOPC and Max Sensitivity scores. The results show that indeed the HiResCAM metric scores are better than the GradCAM ones.

Sources:
- HiResCAM: https://arxiv.org/pdf/2011.08891.pdf
- GradCAM: https://arxiv.org/pdf/1610.02391.pdf
- Dataset: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
