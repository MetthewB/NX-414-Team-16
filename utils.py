### Utils
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from sklearn.metrics import explained_variance_score
from torch.optim.lr_scheduler import StepLR


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


def train_model(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs=10, device='cpu', scheduler=None):
    """
    Train the model and validate it after each epoch.

    Args:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        loss_function (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.

    Returns:
        None
    """
    model.to(device)

    # Learning rate scheduler
    if scheduler is None:
        # Use StepLR scheduler with step size of 5 and gamma of 0.1
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, spikes in train_dataloader:
            images, spikes = images.to(device), spikes.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, spikes)

            # Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, spikes in val_dataloader:
                images, spikes = images.to(device), spikes.to(device)

                # Forward pass
                outputs = model(images)
                loss = loss_function(outputs, spikes)

                val_loss += loss.item()

        # Step the learning rate scheduler
        scheduler.step()

        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader):.4f}, "
              f"Val Loss: {val_loss/len(val_dataloader):.4f}")

        # Stop training if loss becomes NaN or Inf
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
            print("Training stopped due to unstable loss (NaN or Inf).")
            break


def evaluate_model(model, dataloader, spikes_val, device='cpu', prev_model=None, prev_dataloader=None):
    """
    Evaluate the model (and optionally a previous model) using explained variance and correlation.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        spikes_val (numpy array): Ground truth neural activity for validation.
        device (str): Device to evaluate on ('cpu' or 'cuda').
        prev_model (nn.Module, optional): A previous model for comparison.

    Returns:
        None
    """
    model.to(device)
    model.eval()

    overall_ev, overall_corr, ev_per_neuron, corr_per_neuron = compute_ev_and_corr(model, dataloader, spikes_val)

    print(f'[Current Model] Overall explained variance: {overall_ev:.4f}')
    print(f'[Current Model] Overall correlation: {overall_corr:.4f}')

    # If previous model is provided, evaluate it too
    if prev_model is not None:
        prev_model.to(device)
        prev_model.eval()

        prev_ev, prev_corr, prev_ev_per_neuron, prev_corr_per_neuron = compute_ev_and_corr(prev_model, prev_dataloader, spikes_val)

        print(f'[Previous Model] Overall explained variance: {prev_ev:.4f}')
        print(f'[Previous Model] Overall correlation: {prev_corr:.4f}')
    else:
        prev_ev_per_neuron, prev_corr_per_neuron = None, None

    # Plot histograms for explained variance & correlation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(ev_per_neuron, bins=20, alpha=0.6, label='Current Model', color='blue', edgecolor='black')
    if prev_ev_per_neuron is not None:
        axs[0].hist(prev_ev_per_neuron, bins=20, alpha=0.6, label='Previous Model', color='red', edgecolor='black')
    axs[0].set_title('Explained Variance per Neuron')
    axs[0].set_xlabel('Explained Variance')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(corr_per_neuron, bins=20, alpha=0.6, label='Current Model', color='green', edgecolor='black')
    if prev_corr_per_neuron is not None:
        axs[1].hist(prev_corr_per_neuron, bins=20, alpha=0.6, label='Previous Model', color='red', edgecolor='black')
    axs[1].set_title('Correlation per Neuron')
    axs[1].set_xlabel('Correlation Coefficient')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_model_main_comp(model, dataloader, spikes_val, device, 
                             prev_model, prev_dataloader, first_model, first_dataloader):
    """
    Evaluate the model (and optionally a previous model) using explained variance and correlation.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        spikes_val (numpy array): Ground truth neural activity for validation.
        device (str): Device to evaluate on ('cpu' or 'cuda').
        prev_model (nn.Module, optional): A previous model for comparison.

    Returns:
        None
    """
    model.to(device)
    model.eval()
    prev_model.to(device)
    prev_model.eval()
    first_model.to(device)
    first_model.eval()
    

    overall_ev, overall_corr, ev_per_neuron, corr_per_neuron = compute_ev_and_corr(model, dataloader, spikes_val)

    print(f'[Current Model] Overall explained variance: {overall_ev:.4f}')
    print(f'[Current Model] Overall correlation: {overall_corr:.4f}')

    prev_ev, prev_corr, prev_ev_per_neuron, prev_corr_per_neuron = compute_ev_and_corr(prev_model, prev_dataloader, spikes_val)

    print(f'[Previous Model] Overall explained variance: {prev_ev:.4f}')
    print(f'[Previous Model] Overall correlation: {prev_corr:.4f}')

    first_ev, first_corr, first_ev_per_neuron, first_corr_per_neuron = compute_ev_and_corr(first_model, first_dataloader, spikes_val)

    print(f'[First Model] Overall explained variance: {first_ev:.4f}')
    print(f'[First Model] Overall correlation: {first_corr:.4f}')

    # Plot histograms for explained variance & correlation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(ev_per_neuron, bins=20, alpha=0.6, label='efficientnet_b5', color='blue', edgecolor='black')
    # Previous model
    axs[0].hist(prev_ev_per_neuron, bins=20, alpha=0.6, label='resnet50', color='red', edgecolor='black')
    # First model
    axs[0].hist(first_ev_per_neuron, bins=20, alpha=0.6, label='deeper_cnn', color='orange', edgecolor='black')

    axs[0].set_title('Explained Variance per Neuron')
    axs[0].set_xlabel('Explained Variance')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(corr_per_neuron, bins=20, alpha=0.6, label='efficientnet_b5', color='blue', edgecolor='black')
    # Previous model
    axs[1].hist(prev_corr_per_neuron, bins=20, alpha=0.6, label='resnet50', color='red', edgecolor='black')
    # First model
    axs[1].hist(first_corr_per_neuron, bins=20, alpha=0.6, label='deeper_cnn', color='orange', edgecolor='black')
    axs[1].set_title('Correlation per Neuron')
    axs[1].set_xlabel('Correlation Coefficient')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_model_bi(model_1, dataloader, spikes_val, device, model_2, resnet, res_dataloader):
    """
    Evaluate the model using explained variance and correlation.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        spikes_val (numpy array): Ground truth neural activity for validation.
        loss_function (nn.Module, optional): Loss function (if needed for logging).
        device (str): Device to evaluate on ('cpu' or 'cuda').

    Returns:
        None
    """
    model_1.to(device)
    model_1.eval()
    model_2.to(device)
    model_2.eval()
    resnet.to(device)
    resnet.eval()

    # Compute explained variance & correlation
    overall_ev_1, overall_corr_1, ev_per_neuron_1, corr_per_neuron_1 = compute_ev_and_corr_biobj(model_1, dataloader, spikes_val)
    overall_ev_2, overall_corr_2, ev_per_neuron_2, corr_per_neuron_2 = compute_ev_and_corr_biobj(model_2, dataloader, spikes_val)
    resnet_ev, resnet_corr, resnet_ev_per_neuron, resnet_corr_per_neuron = compute_ev_and_corr(resnet, res_dataloader, spikes_val)

    print(f'Overall explained variance biobjective weight 1: {overall_ev_2:.4f}')
    print(f'Overall correlation biobjective weight 1: {overall_corr_2:.4f}')

    print(f'Overall explained variance biobjective weight 0.3: {overall_ev_1:.4f}')
    print(f'Overall correlation biobjective weight 0.3: {overall_corr_1:.4f}')

    print(f'Overall explained variance resnet: {resnet_ev:.4f}')
    print(f'Overall correlation resnet: {resnet_corr:.4f}')

    # Plot histograms for explained variance & correlation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(ev_per_neuron_1, bins=20, color='blue', label='Biobjective 0.3', alpha=0.6, edgecolor='black')
    axs[0].hist(ev_per_neuron_2, bins=20, color='orange', label='Biobjective 1', alpha=0.6, edgecolor='black')
    axs[0].hist(resnet_ev_per_neuron, bins=20, alpha=0.6, label='efficientnet_b5', color='red', edgecolor='black')
    axs[0].set_title('Explained Variance per Neuron')
    axs[0].set_xlabel('Explained Variance')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(corr_per_neuron_1, bins=20, color='blue', label='Biobjective 0.3', alpha=0.6, edgecolor='black')
    axs[1].hist(corr_per_neuron_2, bins=20, color='orange', label='Biobjective 1', alpha=0.6, edgecolor='black')
    axs[1].hist(resnet_corr_per_neuron, bins=20, alpha=0.6, label='efficientnetB5', color='red', edgecolor='black')
    axs[1].set_title('Correlation per Neuron')
    axs[1].set_xlabel('Correlation Coefficient')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def train_model_biobj(model, train_dataloader, val_dataloader, criterion_activity, criterion_label, coef_label, optimizer, epochs=10, device='cpu'):
    """
    Train the model and validate it after each epoch.

    Args:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        loss_function (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.

    Returns:
        None
    """
    model.to(device)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, spikes, objects in train_dataloader:
            images, spikes, objects = images.to(device), spikes.to(device), objects.to(device)

            optimizer.zero_grad()
            # Forward pass
            activity_output, label_output = model(images)

            loss_activity = criterion_activity(activity_output, spikes)
            loss_label = criterion_label(label_output, objects)

            loss = loss_activity + (coef_label*loss_label)

            # Backward pass & optimization
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, spikes, objects in val_dataloader:
                images, spikes, objects = images.to(device), spikes.to(device), objects.to(device)

                # Forward pass
                activity_output, label_output = model(images)
                loss_activity = criterion_activity(activity_output, spikes)
                loss_label = criterion_label(label_output, objects)

                loss = loss_activity + (coef_label*loss_label)

                val_loss += loss.item()

        # Step the learning rate scheduler
        scheduler.step()

        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader):.4f}, "
              f"Val Loss: {val_loss/len(val_dataloader):.4f}")

        # Stop training if loss becomes NaN or Inf
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
            print("Training stopped due to unstable loss (NaN or Inf).")
            break


def evaluate_model_main_comp(model, dataloader, spikes_val, device,
                             prev_model, prev_dataloader, first_model, first_dataloader):
    """
    Evaluate the model (and optionally a previous model) using explained variance and correlation.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        spikes_val (numpy array): Ground truth neural activity for validation.
        device (str): Device to evaluate on ('cpu' or 'cuda').
        prev_model (nn.Module, optional): A previous model for comparison.

    Returns:
        None
    """
    model.to(device)
    model.eval()
    prev_model.to(device)
    prev_model.eval()
    first_model.to(device)
    first_model.eval()


    overall_ev, overall_corr, ev_per_neuron, corr_per_neuron = compute_ev_and_corr(model, dataloader, spikes_val)

    print(f'[Current Model] Overall explained variance: {overall_ev:.4f}')
    print(f'[Current Model] Overall correlation: {overall_corr:.4f}')

    prev_ev, prev_corr, prev_ev_per_neuron, prev_corr_per_neuron = compute_ev_and_corr(prev_model, prev_dataloader, spikes_val)

    print(f'[Previous Model] Overall explained variance: {prev_ev:.4f}')
    print(f'[Previous Model] Overall correlation: {prev_corr:.4f}')

    first_ev, first_corr, first_ev_per_neuron, first_corr_per_neuron = compute_ev_and_corr(first_model, first_dataloader, spikes_val)

    print(f'[First Model] Overall explained variance: {first_ev:.4f}')
    print(f'[First Model] Overall correlation: {first_corr:.4f}')

    # Plot histograms for explained variance & correlation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(ev_per_neuron, bins=20, alpha=0.6, label='efficientnet_b5', color='blue', edgecolor='black')
    # Previous model
    axs[0].hist(prev_ev_per_neuron, bins=20, alpha=0.6, label='resnet50', color='red', edgecolor='black')
    # First model
    axs[0].hist(first_ev_per_neuron, bins=20, alpha=0.6, label='deeper_cnn', color='orange', edgecolor='black')

    axs[0].set_title('Explained Variance per Neuron')
    axs[0].set_xlabel('Explained Variance')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(corr_per_neuron, bins=20, alpha=0.6, label='efficientnet_b5', color='blue', edgecolor='black')
    # Previous model
    axs[1].hist(prev_corr_per_neuron, bins=20, alpha=0.6, label='resnet50', color='red', edgecolor='black')
    # First model
    axs[1].hist(first_corr_per_neuron, bins=20, alpha=0.6, label='deeper_cnn', color='orange', edgecolor='black')
    axs[1].set_title('Correlation per Neuron')
    axs[1].set_xlabel('Correlation Coefficient')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    plt.tight_layout()
    plt.show()