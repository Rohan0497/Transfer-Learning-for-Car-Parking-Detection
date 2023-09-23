from config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_EPOCHS,
    LR,
    MOMENTUM,
    WEIGHT_DECAY,
    GAMMA,
    STEP_SIZE,
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS,
    RESIZE_TO,
    VALID_DIR,
    TRAIN_DIR,
    RESOLUTION_SCHEDULE
)
from model import create_model
from utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    show_transformed_image
)
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR

import torch
import matplotlib.pyplot as plt
import time
import os

plt.style.use('ggplot')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 



# Function for running training iterations.
def train(train_data_loader, model):
    """
    Function to train the model for one epoch.
    
    Parameters:
    - train_data_loader (DataLoader): DataLoader for training data.
    - model (torch.nn.Module): The model to be trained.
    
    Returns:
    - loss_value (float): Loss value for the epoch.
    """
   
    print('Training')
    # Set the model to training mode. This will turn on layers that behave
    # differently during training, such as dropout.
    model.train()
    
    # Initialize a progress bar to give visual feedback about training progress.
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    # Loop through batches of data.
    for i, data in enumerate(prog_bar):
        # Zero the gradients. This clears old gradients from the last step.
        optimizer.zero_grad()
        images, targets = data

        # Move images and targets to the computation device (GPU or CPU).
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        # Forward pass: compute the model's predictions for the batch.
        loss_dict = model(images, targets)

        # Compute the total loss. 
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Send the loss value to our custom averager to compute running average.
        train_loss_hist.send(loss_value)

        # Backward pass: compute the gradient of the loss with respect to model parameters.
        losses.backward()

        # Update the model's parameters.
        optimizer.step()
    
        # Update the progress bar with the current loss value.
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
     # Return the final loss value for this epoch.
    return loss_value

# Function for running validation iterations.
def validate(valid_data_loader, model):

    """
    Function to validate the model for one epoch.
    
    Parameters:
    - valid_data_loader (DataLoader): DataLoader for validation data.
    - model (torch.nn.Module): The trained model.
    
    Returns:
    - target (list): Ground truth annotations.
    - preds (list): Model predictions.
    - metric_summary (dict): Dictionary containing mAP values.
    """

    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # Extract and store predictions and true values for mAP calculation.

        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return target,preds,metric_summary

if __name__ == '__main__':
    # Create an outputs directory to store results.
    os.makedirs('outputs', exist_ok=True)

    # Initialize datasets and dataloaders for training and validation.
    train_dataset = create_train_dataset(TRAIN_DIR)
    valid_dataset = create_valid_dataset(VALID_DIR)
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)


    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the SSD model and move it to the computation device.
    model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)
    model = model.to(DEVICE)
    print(model)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    #Intialising Hyperparameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    # optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=STEP_SIZE, gamma=GAMMA, verbose=True
    )

    # Initialize the averager and lists to keep track of metrics.
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Define the model's name for saving.
    MODEL_NAME = 'model'

    # Visualize some transformed training images.
    if VISUALIZE_TRANSFORMED_IMAGES:
        
        show_transformed_image(train_loader)

    # Initialize the custom class to save the best model..
    save_best_model = SaveBestModel()
    
    
# Initialize early stopping parameters.
EARLY_STOPPING_EPOCHS = 5

# Initialize variables for early stopping
best_map = 0.0
epochs_without_improvement = 0

# Training loop starts here.
for epoch in range(NUM_EPOCHS):
    if epoch in RESOLUTION_SCHEDULE:
        width, height = RESOLUTION_SCHEDULE[epoch]
        train_dataset.set_resolution(width, height)
        valid_dataset.set_resolution(width, height)
        train_loader = create_train_loader(train_dataset, NUM_WORKERS)
        valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
        print(f"Switched to resolution: {width}x{height} at epoch {epoch}")
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

    # Reset the loss history for this epoch.
    train_loss_hist.reset()

  
    # Start timer and Train and validate for this epoch.
    start = time.time()
    train_loss = train(train_loader, model)
    targets,preds,metric_summary = validate(valid_loader, model)
    print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
    print(f"Epoch #{epoch+1} mAP: {metric_summary['map']}")   
    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

     # Add metrics to lists for plotting later.
    train_loss_list.append(train_loss)
    map_50_list.append(metric_summary['map_50'])
    map_list.append(metric_summary['map'])

    # Check for mAP improvement
    # Early stopping check.
    current_map = metric_summary['map']
    if current_map > best_map:
        best_map = current_map
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # If mAP hasn't improved for a specified number of epochs, stop training
    if epochs_without_improvement >= EARLY_STOPPING_EPOCHS:
        print("Early stopping due to mAP plateau!")
        break

    # Save  best model.
    save_best_model(
        model, float(metric_summary['map']), epoch, 'outputs'
    )
    # Save  current model.
    save_model(epoch, model, optimizer)

    # Save and plot  loss.
    save_loss_plot(OUT_DIR, train_loss_list)

    # Save and plot  mAP.
    save_mAP(OUT_DIR, map_50_list, map_list)

    # #Save and plot precision_recall_curve
    # plot_precision_recall_curve(preds, targets, NUM_CLASSES)

    # # Save and plot confusion matrix
    # plot_confusion_matrix(targets, preds,NUM_CLASSES)
    scheduler.step()
