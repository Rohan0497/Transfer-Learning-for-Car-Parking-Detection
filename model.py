import torchvision
import torch.nn as nn
from torchvision.models.detection.ssd import SSDClassificationHead,SSDHead, SSD, DefaultBoxGenerator
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights



def create_model(num_classes=91, size=300):

    """
    Create Base Single Shot MultiBox Detector (SSD) model.
    
    Parameters:
    - num_classes (int): Number of classes for detection.
    - size (int): Size of the input image.
    - nms (float): Non-maximum suppression threshold.
    
    Returns:
    - model (SSD): SSD model.
    """
    # Load the Torchvision pretrained model.
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1
    )
    # Retrieve the list of input channels. 
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))


    anchor_generator1 = DefaultBoxGenerator(
    [[0.8921787], [1.2399989, 1.3902301], [1.765101, 1.9821488], [1.2399989, 1.3902301], [0.8921787], [0.8921787]]
    )
    # List containing number of anchors based on aspect ratios.
    num_anchors = anchor_generator1.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    # Configure the image size for transforms
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

if __name__ == '__main__':
    # Create the model
    model = create_model(2, 640)
    print(model)

    # Calculate and print the total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")