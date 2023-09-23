import torch

from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader
from utils import plot_precision_recall_curve, plot_confusion_matrix

# Evaluation function
def validate(valid_data_loader, model):

    """
    Evaluate the model's performance on the validation dataset.
    
    Parameters:
    - valid_data_loader (DataLoader): DataLoader for validation data.
    - model (torch.nn.Module): The trained model.
    
    Returns:
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
        
        # Model's forward pass.
        with torch.no_grad():
            outputs = model(images, targets)

        # Extract and store predictions and true values for mAP calculation.
        for idx in range(len(images)):
            true_dict = {
                'boxes': targets[idx]['boxes'].detach().cpu(),
                'labels': targets[idx]['labels'].detach().cpu()
            }
            preds_dict = {
                'boxes': outputs[idx]['boxes'].detach().cpu(),
                'scores': outputs[idx]['scores'].detach().cpu(),
                'labels': outputs[idx]['labels'].detach().cpu()
            }
            preds.append(preds_dict)
            target.append(true_dict)
   

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return target,preds,metric_summary

if __name__ == '__main__':
    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES, size=640)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset(
        '/home/ec22302/msc_project/Pytorch_SSD_test/data/test'
    )
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    targets,preds,metric_summary = validate(test_loader, model)
    print(f"mAP_50: {metric_summary['map_50']*100:.3f}")
    print(f"mAP_50_95: {metric_summary['map']*100:.3f}")

    # #Save and plot precision_recall_curve
    # plot_precision_recall_curve(preds, targets, NUM_CLASSES)

    # # Save and plot confusion matrix
    # plot_confusion_matrix(targets, preds,NUM_CLASSES)