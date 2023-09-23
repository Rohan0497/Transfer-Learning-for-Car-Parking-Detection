import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES,VISUALIZE_TRANSFORMED_IMAGES
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
from sklearn.metrics import confusion_matrix, precision_recall_curve
plt.style.use('ggplot')

class Averager:
    """
    A class to maintain the running average of a quantity.
    
    Attributes:
        current_total (float): Current total of the values.
        iterations (float): Number of values added.
    """
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value: float):
        """
        Add a new value to the running total.
        
        Args:
            value (float): The new value to be added.
        """
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self) -> float:
        """
        Compute the current average of the values.
        
        Returns:
            float: Current average value.
        """
        return 0 if self.iterations == 0 else self.current_total / self.iterations
    
    def reset(self):
        """
        Reset the running total and the number of values.
        """
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    A utility class to save the best model based on validation mAP.
    
    Attributes:
        best_valid_map (float): The highest validation mAP observed so far.
    """
    def __init__(self, best_valid_map: float = 0.0):
        self.best_valid_map = best_valid_map
        
    def __call__(self, model, current_valid_map: float, epoch: int, OUT_DIR: str):
        """
        Save the model if the current validation mAP is better than the best observed so far.
        
        Args:
            model: The model to be saved.
            current_valid_map (float): The current validation mAP.
            epoch (int): The current training epoch.
            OUT_DIR (str): The directory where the model should be saved.
        """
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
            }, f"{OUT_DIR}/best_model.pth")



def collate_fn(batch):
    """
    Custom collate function to handle possible errors due to image transformations.

    Args:
    - batch (list): A list of tuples where each tuple contains an image and its associated targets.

    Returns:
    - list: Processed images.
    - list: Corresponding targets for each image.

    """
    return tuple(zip(*batch))

# Define the training tranforms.
def get_train_transform():
    """
    Get transformations to be applied on training images.

    Returns:
    - A.Compose: Albumentations composed transformations for training.
    """
    return A.Compose([
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        A.RandomGamma(p=0.3),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', 
                                label_fields=['labels']))
   

# Define the validation transforms.
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc',
                                 label_fields=['labels']))



def show_transformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the transformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    num_images = 5
    if len(train_loader) > 0:
        image_indices = random.sample(range(len(train_loader.dataset)), num_images)
        for idx in image_indices:
            image, target = train_loader.dataset[idx]
            image = image.permute(1, 2, 0).cpu().numpy()
            boxes = target['boxes'].cpu().numpy().astype(np.int32)
            labels = target['labels'].cpu().numpy().astype(np.int32)

            # Using matplotlib for visualization
            
            # Displaying image without bounding boxes
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis('off')
            if VISUALIZE_TRANSFORMED_IMAGES:
                output_dir = "outputs/transformed"
                os.makedirs(output_dir, exist_ok=True)
                filename = f"without_bbox_{idx}.png"
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
                plt.show()

            # Displaying image with bounding boxes
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            for box_num, box in enumerate(boxes):
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                         linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(box[0], box[1]-5, CLASSES[labels[box_num]], 
                         bbox=dict(facecolor='red', alpha=0.5), fontsize=10, color='white')
            
            plt.axis('off')
            if VISUALIZE_TRANSFORMED_IMAGES:
                output_dir = "outputs/transformed"
                os.makedirs(output_dir, exist_ok=True)
                filename = f"with_bbox_{idx}.png"
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
                plt.show()


def save_model(epoch: int, model, optimizer):
    """
    Save the state of the training model and optimizer.
    
    Args:
        epoch (int): Current training epoch.
        model: The model to be saved.
        optimizer: The optimizer to be saved.
    """
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'outputs/last_model.pth')

def save_loss_plot(OUT_DIR: str, train_loss_list: list, x_label: str = 'epochs', y_label: str = 'train loss', save_name: str = 'train_loss'):
    """
    Save the training loss plot.
    
    Args:
        OUT_DIR (str): Directory to save the plot.
        train_loss_list (list): List of training loss values.
        x_label (str, optional): X-axis label. Defaults to 'epochs'.
        y_label (str, optional): Y-axis label. Defaults to 'train loss'.
        save_name (str, optional): Filename for the plot. Defaults to 'train_loss'.
    """
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
    print('SAVING PLOTS COMPLETE.')

def save_mAP(OUT_DIR: str, map_05: list, map: list):
    """
    Save the mAP@0.5 and mAP@0.5:0.95 plots.
    
    Args:
        OUT_DIR (str): Directory to save the plots.
        map_05 (list): List of mAP values at 0.5 IoU.
        map (list): List of mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-', 
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-', 
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/map.png")



def plot_precision_recall_curve(preds, targets, num_classes=3):
    """
    Plot the precision-recall curve for each class excluding the background class.
    
    Parameters:
    - preds (list): List of dictionaries containing 'scores' and 'labels'.
    - targets (list): List of dictionaries containing 'labels'.
    - num_classes (int, optional): Total number of classes including background. Defaults to 3.
    
    Note:
    This function saves the precision-recall curve in the "outputs/" directory.
    """
    
    # Initialize lists to store precision and recall values for each class.
    precision_list = []
    recall_list = []
    
    # Skip the background class (0) and calculate precision and recall for the remaining classes.
    for class_id in range(1, num_classes):  # Start from 1 to exclude background class
        y_true = []
        y_score = []
        
        for i in range(len(preds)):
            # Check if 'scores' key is present in the dictionary.
            if 'scores' not in preds[i]:
                continue
            
            mask = preds[i]['labels'] == class_id
            scores = preds[i]['scores'][mask].numpy()
            binary_true_labels = (targets[i]['labels'] == class_id).float().numpy()
            
            # If there are no true or predicted labels for this class, skip.
            if len(binary_true_labels) == 0 or len(scores) == 0:
                continue
            
            # Ensure that the lengths of scores and labels are consistent.
            min_length = min(len(binary_true_labels), len(scores))
            y_true.extend(binary_true_labels[:min_length])
            y_score.extend(scores[:min_length])
        
        # If y_true or y_score is empty, skip this class.
        if len(y_true) == 0 or len(y_score) == 0:
            continue

        # Calculate precision and recall for binary classification (current class vs all others).
        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
        precision_list.append(precision)
        recall_list.append(recall)
    
    # Plot precision-recall curve for the non-background classes.
    plt.figure(figsize=(10, 7))
    class_names = ['space-occupied', 'space-emptied']  # Names for the non-background classes
    for idx, class_name in enumerate(class_names):
        if idx < len(precision_list):  # Check if precision and recall values were computed for this class
            plt.plot(recall_list[idx], precision_list[idx], label=class_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.grid(True)


    plt.savefig(f"outputs/precision_recall_curve.png")




def plot_confusion_matrix(y_true, y_pred,num_classes= 3):
    """
    Plot the confusion matrix for the given true and predicted labels.
    
    Parameters:
    - y_true (list): List of dictionaries containing true 'labels'.
    - y_pred (list): List of dictionaries containing predicted 'labels'.
    - num_classes (int, optional): Total number of classes including background. Defaults to 3.
    
    Returns:
    - cm (numpy.ndarray): The computed confusion matrix.
    
    Note:
    This function saves the confusion matrix in the "outputs/" directory and also displays it.
    """
    # Flatten the labels and predictions
    true_labels = []
    pred_labels = []
    
    for i in range(len(y_pred)):
        # Check if 'labels' key is present in the dictionary.
        if 'labels' not in y_pred[i]:
            continue
        
        # For the current image/sample
        true_cls = y_true[i]['labels'].numpy()
        pred_cls = y_pred[i]['labels'].numpy()

        # If there are no true or predicted labels for this sample, skip.
        if len(true_cls) == 0 or len(pred_cls) == 0:
            continue
        
        # Ensure that the lengths of true_cls and pred_cls are consistent.
        min_length = min(len(true_cls), len(pred_cls))
        true_labels.extend(true_cls[:min_length])
        pred_labels.extend(pred_cls[:min_length])

    # Compute the confusion matrix for labels 1 and 2
    cm = confusion_matrix(true_labels, pred_labels, labels=[1, 2])
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    
    # Use the actual size of the confusion matrix to set ticks
    tick_marks = np.arange(cm.shape[0])
    class_names = ['space empty', 'space occupied']
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Annotate the confusion matrix with the actual values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"outputs/confusion_matrix.png")
    plt.show()
    
    return cm

