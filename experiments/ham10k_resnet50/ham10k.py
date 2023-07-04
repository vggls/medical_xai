'''
As long as HAM10000 is downloaded into Google Colab from Kaggle you may use the folloing methods to 
organize the files into training, validation and test set of 80%, 10% and 10% instances per class.

Methods:
    - map_images_to_labels: adds an additional column named 'image_path' to the metadata dataframe
    - create_datasets: creates train/val/test sets with the torchvision.datasets.ImageFolder method
                        Each class contributes 60%, 20% and 20% to the train/val/test sets and the sets are people independent
    - summaries: returns class distribution per train, validation and test set

example code :
dataframe = map_images_to_labels()
train_dataset, validation_dataset, test_dataset= create_datasets(dataframe, train_transforms, test_transforms)
labels = train_dataset.class_to_idx
summaries(labels, bar_plot=True)
'''
import os
import pandas as pd
import shutil
import plotly.graph_objects as go
from torchvision import datasets

def map_images_to_paths():
    
    dictionary = {}
    for folder_path in ['HAM10000_images_part_1/', 'HAM10000_images_part_2/']:
        for image in os.listdir(folder_path):
            image_id = image.replace(".jpg", "")
            image_path = folder_path + image
            dictionary[image_id] = image_path

    dataframe = pd.read_csv('HAM10000_metadata.csv')
    dataframe['image_path'] = dataframe['image_id'].map(dictionary)

    return dataframe


def create_datasets(dataframe, train_transforms, test_transforms):
    
    classes = dataframe['dx'].unique()         #classes
    
    # paths
    training_path = './training_dataset/'
    validation_path = './validation_dataset/'
    test_path = './test_dataset/'

    # dictionary that contains the total numbet of instances per class
    totals = {}
    totals = {key: None for key in classes}
    for clas in classes:
        totals[clas]=len(dataframe[dataframe['dx']==clas])

    # dictionary to track no of images per set and class
    training_imgs = {key: 0 for key in classes}
    validation_imgs = {key: 0 for key in classes}
    test_imgs = {key: 0 for key in classes}

    # list that trachs the set (one only) each person contributes to
    training_people = []
    validation_people = []
    test_people = []

    # create folders
    for clas in classes:
        os.makedirs(training_path + clas)
        os.makedirs(validation_path + clas)
        os.makedirs(test_path + clas)

    sorted = dataframe.sort_values(by=['lesion_id'])

    
    lesions = sorted['lesion_id'].unique()   #people

    pending = []

    # copy files to folders
    for i, lesion in enumerate(lesions):

        lesion_df = sorted[sorted['lesion_id']==lesion]
        
        for image_index in lesion_df.index:                   # iterate over person images 
            clas = lesion_df.loc[image_index,'dx']            # image class
            source = lesion_df.loc[image_index,'image_path']  # image source

            if (training_imgs[clas]<=0.6 * totals[clas]) and (lesion not in validation_people) and (lesion not in test_people):
                destination = training_path + clas
                training_people.append(lesion)
                training_imgs[clas]+=1
            elif (validation_imgs[clas]<=0.2 * totals[clas]) and (lesion not in training_people) and (lesion not in test_people):
                destination = validation_path + clas
                validation_people.append(lesion)
                validation_imgs[clas]+=1
            elif (lesion not in training_people) and (lesion not in validation_people):
                destination = test_path + clas
                test_people.append(lesion)
                test_imgs[clas]+=1
            else:
                pending.append([i,lesion,clas,source])
            shutil.copy(source, destination)

    assert len(list(set(training_people) & set(test_people)))==0
    assert len(list(set(training_people) & set(validation_people)))==0
    assert len(list(set(validation_people) & set(test_people)))==0

    if  len(pending)!=0: #do not know why there are pendings; there are 2
        #print(len(pending)) #2
        for img in pending:
            source = img[3]
            if img[1] in training_people:
                destination = training_path + img[2]
                training_imgs[img[2]]+=1
            elif img[1] in validation_people:
                destination = validation_path + img[2]
                validation_imgs[img[2]]+=1
            elif img[1] in test_people:
                destination = test_path + img[2]
                test_imgs[img[2]]+=1
            shutil.copy(source, destination)

    # ImageFolder method
    train_dataset = datasets.ImageFolder(root=training_path, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(root=validation_path, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)
          
    return train_dataset, validation_dataset, test_dataset


def summaries(labels, bar_plot=True):

    # Note that we do not calculate the summaries from the created datasets variables 
    # It is much faster if we calculate the number of contents of the folders instead
    
    for set, path in zip(['Training dataset', 'Validation dataset', 'Test dataset'],
                            ['./training_dataset/', './validation_dataset/', './test_dataset/']):

        no_of_set_images = []
        for clas in labels.keys():
            class_images = os.listdir(path + clas)
            no_of_set_images.append(len(class_images))
        
        pairs = ['{} (label {})'.format(k,v) for k, v in labels.items()]

        if bar_plot:
            fig = go.Figure([go.Bar(x=no_of_set_images, y=pairs, 
                                    text=no_of_set_images, textposition='outside',
                                    orientation='h')])
            fig.update_layout(width=850, height=420, title='{} : Class distribution'.format(set), title_x=0.5)
            fig.show()
        else:
            print(f'{set}')
            print([(x,y) for x,y in zip(pairs,no_of_set_images)])
            print('\n')