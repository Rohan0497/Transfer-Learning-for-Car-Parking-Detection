import torchvision
import torch.nn as nn
from torchvision.models.detection.ssd import SSDClassificationHead, SSDHead, SSD, DefaultBoxGenerator
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights

def create_model(num_classes=2, size=640):
    """
    Create SSD model for car park detection.
    """
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    anchor_generator = DefaultBoxGenerator(
        [[0.8921787], [1.2399989, 1.3902301], [1.765101, 1.9821488], [1.2399989, 1.3902301], [0.8921787], [0.8921787]]
    )
    num_anchors = anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

if __name__ == "__main__":
    model = create_model(num_classes=2)
    print("Model initialized successfully.")