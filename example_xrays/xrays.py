
"""
Custom written method that takes the X-rays data, once downloaded from Kaggle to Colab's 'content' folder,
and creates training_dataset, validation_dataset and test_dataset variables by the torchvision datasets.ImageFolder method.
Each set contains 80%, 10% and 10% instances per class respectively.

example code:
train_dataset, validation_dataset, test_dataset= create_datasets(train_transforms, test_transforms)
labels = train_dataset.class_to_idx
summaries(labels, bar_plot=True)
"""
import os
import shutil
from torchvision  import datasets
import plotly.graph_objects as go


def create_datasets(train_transforms, 
                    test_transforms):
    
    datapath = './COVID-19_Radiography_Dataset/'
    
    # ----- create empty folders --------------------------------------------------------------
    os.makedirs(datapath + 'training_dataset/COVID')
    os.makedirs(datapath + 'training_dataset/Lung_Opacity')
    os.makedirs(datapath + 'training_dataset/Normal')
    os.makedirs(datapath + 'training_dataset/Viral Pneumonia')
    
    os.makedirs(datapath + 'validation_dataset/COVID')
    os.makedirs(datapath + 'validation_dataset/Lung_Opacity')
    os.makedirs(datapath + 'validation_dataset/Normal')
    os.makedirs(datapath + 'validation_dataset/Viral Pneumonia')
    
    os.makedirs(datapath + 'test_dataset/COVID')
    os.makedirs(datapath + 'test_dataset/Lung_Opacity')
    os.makedirs(datapath + 'test_dataset/Normal')
    os.makedirs(datapath + 'test_dataset/Viral Pneumonia')
    
    # ----- copy files to the folders ---------------------------------------------------------
    classes = [ 'COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

    for clas in classes:
    
        class_path = './COVID-19_Radiography_Dataset/{}/images'.format(clas)
        class_imgs = os.listdir(class_path)
        sorted_class_imgs = sorted(class_imgs) # to ensure that each time we work with the same train, val and test data
        h = 1
        for img in sorted_class_imgs :
            source = os.path.join (class_path, img) # source = img_path
            if h <= int(len(class_imgs)*0.8) :
                # copy the 80% of the images to the training_dataset
                destination = './COVID-19_Radiography_Dataset/training_dataset/{}'.format(clas)
            elif h <= int(len(class_imgs)*0.9) :
                # copy the 10% of the images to the validation_dataset
                destination = './COVID-19_Radiography_Dataset/validation_dataset/{}'.format(clas)
            else :
                # copy the rest 10% of the images to the test_dataset
                destination = './COVID-19_Radiography_Dataset/test_dataset/{}'.format(clas)
            shutil.copy(source, destination)
            h += 1
            
    # ----- paths -----------------------------------------------------------------------------
    train_imgs_path = datapath + 'training_dataset'
    val_imgs_path = datapath + 'validation_dataset'
    test_imgs_path = datapath + 'test_dataset'
      
    # ----use ImageFolder to create the dataset variables ------------------------------------
    train_dataset = datasets.ImageFolder(root=train_imgs_path, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(root=val_imgs_path, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=test_imgs_path, transform=test_transforms)
      
    return train_dataset, validation_dataset, test_dataset

def summaries(labels, bar_plot=True):

    # Note that we do not calculate the summaries from the created datasets variables 
    # It is much faster if we calculate the number of contents of the folders instead
    
    for set, path in zip(['Training dataset', 'Validation dataset', 'Test dataset'],
                            ['./COVID-19_Radiography_Dataset/training_dataset/', 
                             './COVID-19_Radiography_Dataset/validation_dataset/', 
                             './COVID-19_Radiography_Dataset/test_dataset/']):

        no_of_set_images = []
        for clas in labels.keys():
            class_images = os.listdir(path + clas)
            no_of_set_images.append(len(class_images))
        
        pairs = ['{} (label {})'.format(k,v) for k, v in labels.items()]

        if bar_plot:
            fig = go.Figure([go.Bar(x=no_of_set_images, y=pairs, 
                                    text=no_of_set_images, textposition='outside',
                                    orientation='h')])
            fig.update_layout(width=900, height=500, 
                              title='{} : Class distribution'.format(set), title_x=0.5,
                              bargap = 0.5)
            fig.show()
        else:
            print(f'{set}')
            print([(x,y) for x,y in zip(pairs,no_of_set_images)])
            print('\n')