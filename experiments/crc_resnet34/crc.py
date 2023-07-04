"""
- Load the 'NCT-CRC-HE-100K' and 'CRC-VAL-HE-7K' zipped files in Google Drive.
- Mount Google Drive in Google Colab
- unzip the files in Colab
- Then the below methods perform as follows:
    - create_datasets: Data preparation and torchvision.datasets.ImageFolder method application
                        training_dataset and validation_dataset are 80% and 20% per class of the NCT-CRC-HE-100K.zip file
                        test_dataset is the CRC-VAL-HE-7K.zip file
    - summaries: returns class distribution per train, validation and test set
"""
import os
import shutil
import plotly.graph_objects as go
from torchvision import datasets

def create_datasets(train_transforms, test_transforms):

    classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    
    data_path = './NCT-CRC-HE-100K/'
    training_path = './training_dataset/'
    validation_path = './validation_dataset/'
    test_path = './CRC-VAL-HE-7K/'
    
    # create folders
    for clas in classes:
        os.makedirs(training_path + clas)
        os.makedirs(validation_path + clas)
    
    # copy files to folders
    for clas in classes:
    
        class_imgs = os.listdir(data_path + clas)
        sorted_class_imgs = sorted(class_imgs)
    
        h = 1
        for img in sorted_class_imgs: 
            source = os.path.join(data_path + clas, img)
            if h<=int(len(class_imgs) * 0.8):
                destination= training_path + clas
            else:
                destination = validation_path + clas
            shutil.copy(source, destination)
            h += 1
    
    # ImageFolder method
    train_dataset = datasets.ImageFolder(root=training_path, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(root=validation_path, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)
    
    return train_dataset, validation_dataset, test_dataset

def summaries(bar_plot=True):

    classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    for set, path in zip(['Training dataset', 'Validation dataset', 'Test dataset'],
                            ['./training_dataset/', './validation_dataset/', './CRC-VAL-HE-7K/']):
        no_of_set_images = []
        for clas in classes:
            class_images = os.listdir(path + clas)
            no_of_set_images.append(len(class_images))
        
        if bar_plot:
            fig = go.Figure([go.Bar(x=classes, y=no_of_set_images, 
                                    text=no_of_set_images, textposition='outside')])
            fig.update_layout(width=900, height=550, 
                              title='{} : Class distribution'.format(set), title_x=0.5,
                              bargap = 0.5)
            fig.show()
        else:
            print(f'{set}')
            print([(x,y) for x,y in zip(classes,no_of_set_images)])
            print('\n')