### Utils
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from sklearn.metrics import explained_variance_score


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


def compute_corr(true_vals, preds, divide=False):
    """
    Compute the correlation between true and predicted values.

    Args:
        true_vals (numpy array): Ground truth values.
        preds (numpy array): Predicted values.
        divide (bool): If True, compute correlation per neuron.

    Returns:
        numpy array or float: Correlation values.
    """
    true_vals = np.array(true_vals)
    preds = np.array(preds)

    if divide:
        correlations = []
        for i in range(true_vals.shape[1]):
            std_true = np.std(true_vals[:, i])
            std_pred = np.std(preds[:, i])
            if std_true == 0 or std_pred == 0:
                correlations.append(np.nan)  # Skip neurons with zero std
            else:
                corr = np.corrcoef(true_vals[:, i], preds[:, i])[0, 1]
                correlations.append(corr)
        return np.array(correlations)
    else:
        valid_neurons = []
        for i in range(true_vals.shape[1]):
            std_true = np.std(true_vals[:, i])
            std_pred = np.std(preds[:, i])
            if std_true != 0 and std_pred != 0:
                valid_neurons.append(i)
        if len(valid_neurons) == 0:
            raise ValueError("No valid neurons with non-zero standard deviation.")
        valid_corr = np.corrcoef(true_vals[:, valid_neurons], preds[:, valid_neurons], rowvar=False)
        return np.mean(np.diag(valid_corr[:len(valid_neurons), len(valid_neurons):]))


def compute_ev_and_corr(model, dataloader, spikes_val):
    """
    Compute explained variance and correlation for the model predictions.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        spikes_val (numpy array): Ground truth neural activity for validation.

    Returns:
        tuple: Overall explained variance and correlation.
    """
    model.eval()
    val_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            val_preds.append(outputs.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)

    # Compute explained variance
    ev = explained_variance_score(spikes_val, val_preds, multioutput='raw_values')
    overall_ev = np.mean(ev)

    # Compute correlation
    corr = compute_corr(spikes_val, val_preds, divide=True)
    overall_corr = compute_corr(spikes_val, val_preds)

    return overall_ev, overall_corr, ev, corr