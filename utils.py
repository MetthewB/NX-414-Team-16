### Utils
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle


def load_it_data(path_to_data):
    """ Load IT data

    Args:
        path_to_data (str): Path to the data

    Returns:
        np.array (x6): Stimulus train/val/test; objects list train/val/test; spikes train/val
    """

    datafile = h5py.File(os.path.join(path_to_data,'IT_data.h5'), 'r')

    stimulus_train = datafile['stimulus_train'][()]
    spikes_train = datafile['spikes_train'][()]
    objects_train = datafile['object_train'][()]
    
    stimulus_val = datafile['stimulus_val'][()]
    spikes_val = datafile['spikes_val'][()]
    objects_val = datafile['object_val'][()]
    
    stimulus_test = datafile['stimulus_test'][()]
    objects_test = datafile['object_test'][()]

    ### Decode back object type to latin
    objects_train = [obj_tmp.decode("latin-1") for obj_tmp in objects_train]
    objects_val = [obj_tmp.decode("latin-1") for obj_tmp in objects_val]
    objects_test = [obj_tmp.decode("latin-1") for obj_tmp in objects_test]

    return stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val


def visualize_img(stimulus,objects,stim_idx):
    """Visualize image given the stimulus and corresponding index and the object name.

    Args:
        stimulus (array of float): Stimulus containing all the images
        objects (list of str): Object list containing all the names
        stim_idx (int): Index of the stimulus to plot
    """    
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]

    img_tmp = np.transpose(stimulus[stim_idx],[1,2,0])

    ### Go back from normalization
    img_tmp = (img_tmp*normalize_std + normalize_mean) * 255

    plt.figure()
    plt.imshow(img_tmp.astype(np.uint8),cmap='gray')
    plt.title(str(objects[stim_idx]))
    plt.show()
    return


def list_to_classes(objects):
    '''
    Transform the input list of objects into one of 8 classes: animals, boats, cars, chairs, faces, fruits, planes, or tables.
    Args:
        objects (list of str): A list of object names, where each name corresponds to one of the 64 classes.  
    Returns:
        transformed_objects (list of str): A list of object names, where each name corresponds to one of the 8 classes.
    '''

    fruits = ['apple', 'apricot', 'peach', 'pear', 'raspberry', 'strawberry', 'walnut', 'watermelon']
    animals = ['bear', 'cow', 'dog', 'elephant', 'gorilla', 'hedgehog', 'lioness', 'turtle']
    objects = ['face' if 'face' in s else s for s in objects]
    objects = ['car' if 'car' in s else s for s in objects]
    objects = ['chair' if 'chair' in s else s for s in objects]
    objects = ['airplane' if 'airplane' in s else s for s in objects]
    objects = ['table' if 'table' in s else s for s in objects]
    objects = ['fruit' if s in fruits else s for s in objects]
    objects = ['animal' if s in animals else s for s in objects]
    objects = ['ship' if 'ship' in s else s for s in objects]
    return objects


def classes_to_int(objects):
    """
    Transforms a list of object class names into a list of integers using a pre-defined mapping.
    
    Args:
        objects (list of str): List of object class names.

    Returns:
        objects_int (list of int): List of object class integers based on the pre-defined mapping.
    """
    # define a dictionary that maps the classes to integers
    classes = {'animal':0, 'ship':1, 'car':2, 'chair':3, 'face':4, 'fruit':5, 'airplane':6, 'table':7}
    # create a list of integers
    objects_int = [classes[object] for object in objects]  
    
    return objects_int


def compute_corr(true_vals, preds, divide = False):
    """ Returns the overall correlation between real and predicted values in case the 
    number of neurons under study is 168.
    
    Args:
        true_vals (array of float):  true values 
        output_folder (array of float): pred values

    Returns:
        overall correlation coefficient
    """
    if divide:
        corr = np.diag(np.corrcoef(true_vals, preds, rowvar = False)[:168, 168:])
    else:
        corr = np.mean(np.diag(np.corrcoef(true_vals, preds, rowvar = False)[:168, 168:]))
    
    return corr


def load_pickle_dict(file_path):
    """ Load the pkl dictionary into memory
    
    Args: 
        file_path (string): file path were the pickle dictionary is stored
    
    Returns:
        dictionary of values
    """
    
    with open(file_path, "rb") as f:
        file = pickle.load(f)
    return file