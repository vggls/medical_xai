'''
Source: https://github.com/fastai/imagenette

-download dataset
-convert tgz file to zip
-upload zip file on google drive
-unzip in google colab
-call create_datasets method
'''

import os
import shutil
from torchvision import datasets
import plotly.graph_objects as go


def create_datasets(train_transforms, test_transforms):

    # as per order downloaded
    class_ids = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 
                'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']

    train_val_data_path = './imagenette2/train/'
    training_path = './training_dataset/'
    validation_path = './validation_dataset/'
    test_path = './imagenette2/val/'

    # create folders
    for id in class_ids:
        os.makedirs(training_path + id)
        os.makedirs(validation_path + id)

    # copy files to folders
    for id in class_ids:

        class_imgs = os.listdir(train_val_data_path + id)
        sorted_class_imgs = sorted(class_imgs)

        h = 1
        for img in sorted_class_imgs: 
            source = os.path.join(train_val_data_path + id, img)
            if h<=int(len(class_imgs) * 0.6):
                destination= training_path + id
            else:
                destination = validation_path + id
            shutil.copy(source, destination)
            h += 1

    # ImageFolder method
    train_dataset = datasets.ImageFolder(root=training_path, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(root=validation_path, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)

    return train_dataset, validation_dataset, test_dataset


def summaries(bar_plot=True):

    folders = {'tench': 'n01440764',
              'English springer': 'n02102040',
              'cassette player': 'n02979186',
              'chain saw': 'n03000684',
              'church': 'n03028079',
              'French horn': 'n03394916',
              'garbage truck': 'n03417042',
              'gas pump': 'n03425413',
              'golf ball': 'n03445777',
              'parachute': 'n03888257'}

    for set, path in zip(['Training dataset', 'Validation dataset', 'Test dataset'],
                        ['./training_dataset/', './validation_dataset/', './imagenette2/val/']):

        no_of_set_images = []
        for clas in folders.values():
            class_images = os.listdir(path + clas)
            no_of_set_images.append(len(class_images))
        
        if bar_plot:
            fig = go.Figure([go.Bar(x=list(folders.keys()), y=no_of_set_images, 
                                    text=no_of_set_images, textposition='outside')])
            fig.update_layout(width=900, height=550, 
                              title='{} : Class distribution'.format(set), title_x=0.5,
                              bargap = 0.5)
            fig.show()
        else:
            print(f'{set}')
            print([(x,y) for x,y in zip(folders.keys(),no_of_set_images)])
            print('\n')