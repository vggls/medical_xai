
import os
import shutil
from torchvision  import datasets
import plotly.graph_objects as go

from breakhis_slide_allocation import allocate_slides_to_datasets #custom written code


# ----- create datasets method ----------------------------------------------------------------
def create_datasets(train_transforms, test_transforms):
    
    classes = ['benign', 'malignant']
    
    datapath = '/content/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/'

    magnitudes = ['/100X/', '/200X/', '/400X/', '/40X/']
    
    # ----- create empty folders -------------------
    for clas in classes:
        os.makedirs('training_dataset/' + clas)
        os.makedirs('validation_dataset/' + clas)
        os.makedirs('test_dataset/' + clas)
    
    # ----- slides and datapaths ---------------------------------------------
    benign_slides, malignant_slides = allocate_slides_to_datasets()
    data_paths = ['training_dataset/', 'validation_dataset/', 'test_dataset/']
    
    # ----- copy slide images ------------------------------------------------
    def copy_slide(disease_path, slide, data_path, clas):
        destination = data_path + clas
        for magnitude in magnitudes:
            disease_magn_path = disease_path + slide + magnitude
            for img in os.listdir(disease_magn_path):
                img_source_path = os.path.join (disease_magn_path, img)
                shutil.copy(img_source_path, destination)
    
    # ----- bening slides ----------------------------------------------------
    for (slides, data_path) in zip(benign_slides, data_paths):
        for slide in slides:
            if slide[6]=='A':
                disease_path = datapath + 'benign/SOB/adenosis/'
            elif slide[6]=='F':
                disease_path = datapath + 'benign/SOB/fibroadenoma/'
            elif slide[6]=='P':
                disease_path = datapath + 'benign/SOB/phyllodes_tumor/'
            elif slide[6]=='T':
                disease_path = datapath + 'benign/SOB/tubular_adenoma/'
            copy_slide(disease_path, slide, data_path, 'benign')
    
    # ----- malignant slides -------------------------------------------------
    for (slides, data_path) in zip(malignant_slides, data_paths):
        for slide in slides:
            if slide[6]=='L':
                disease_path = datapath + 'malignant/SOB/lobular_carcinoma/'
            elif slide[6]=='M':
                disease_path = datapath + 'malignant/SOB/mucinous_carcinoma/'
            elif slide[6]=='P':
                disease_path = datapath + 'malignant/SOB/papillary_carcinoma/'
            elif slide[6]=='D':
                disease_path = datapath + 'malignant/SOB/ductal_carcinoma/'
            copy_slide(disease_path, slide, data_path, 'malignant')
    
    # ----- paths ------------------------------------------------------------
    train_imgs_path = 'training_dataset'
    val_imgs_path = 'validation_dataset'
    test_imgs_path = 'test_dataset'
    
    # ----use ImageFolder to create the dataset variables --------------------
    train_dataset = datasets.ImageFolder(root=train_imgs_path, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(root=val_imgs_path, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=test_imgs_path, transform=test_transforms)
    
    return train_dataset, validation_dataset, test_dataset

# ----- training, validation, test datasets class distribution --------------------------------
def summaries(classes, bar_plot=True):

    # Note that we do not calculate the summaries from the created datasets variables 
    # It is much faster if we calculate the number of contents of the folders instead
    
    for set, path in zip(['Training dataset', 'Validation dataset', 'Test dataset'],
                            ['training_dataset/', 
                             'validation_dataset/', 
                             'test_dataset/']):

        no_of_set_images = []
        for clas in classes.keys():
            class_images = os.listdir(path + clas)
            no_of_set_images.append(len(class_images))
        
        pairs = ['{} (label {})'.format(k,v) for k, v in classes.items()]

        if bar_plot:
            fig = go.Figure([go.Bar(x=pairs, y=no_of_set_images, 
                                    text=no_of_set_images, textposition='outside',
                                    orientation='v')])
            fig.update_layout(width=400, height=500, 
                              title='{} : Class distribution'.format(set), title_x=0.5,
                              bargap = 0.5)
            fig.show()
        else:
            print(f'{set}')
            print([(x,y) for x,y in zip(pairs,no_of_set_images)])
            print('\n')
