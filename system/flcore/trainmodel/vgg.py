"""FedLoad VGG11/VGG16 adapted to 32x32 images, matching MininetFed.

Eight/thirteen convolutions plus three linear layers; no batch normalization.
The classifier receives 512 features after five pools, rather than ImageNet's
512*7*7. Hidden linear layers retain the VGG widths of 4096 with dropout 0.5.
"""
from flcore.trainmodel.mininet_models import MininetCNN


class VGG11(MininetCNN):
    optimizer_kind = 'adam'

    def __init__(self, num_classes=10):
        super().__init__('cifar10', num_classes=num_classes)


class VGG16(MininetCNN):
    optimizer_kind = 'adam'

    def __init__(self, num_classes=10):
        super().__init__('cinic10', num_classes=num_classes)
