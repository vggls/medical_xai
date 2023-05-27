## Datasets
- [CRC](https://zenodo.org/record/1214456#.ZHIpcHZBzIU)
- [Covid-19 Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis)
- [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Models
For the purposes of our experiments we used pre-trained ResNet34, ResNet50 and
VGG19 architectures. All networks were customized to the *Conv - Flatten - Class Scores*
structure such that HiResCAM has faithful behaviour when calculated
with respect to the last convolutional layer. 

We note that in the provided experiment code files, the metrics are calculated by loading the entire learned models from a local storage unit.
For the purposes of this repository we provide all model learned weights instead of the entire model.
Thus, in order to run the experiments, one should adjust the learned weights (*learned_weights* folder) to the provided model architectures (*models* folder) instead.
