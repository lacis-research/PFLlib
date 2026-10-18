"""FedLoad VGG11/VGG16 adapted to 32x32 images, matching MininetFed.

Eight/thirteen convolutions plus three linear layers; no batch normalization.
The dense stack receives 512 features after five pools, rather than ImageNet's
512*7*7. Hidden linear layers have width 4096 with dropout 0.5; the final
classifier receives the 4096-dimensional penultimate representation.
"""
from flcore.trainmodel.mininet_models import MininetCNN


class _SemanticVGG(MininetCNN):
    """Features after the second hidden ReLU/dropout, before the classifier.

    The property does not register a second module: checkpoint keys and global
    pruning coordinates remain exclusively under ``layers.*``.
    """

    @property
    def fc(self):
        return self.layers[-1]

    def forward_features(self, x):
        x = self.augment_inputs(x)
        for layer in list(self.layers.children())[:-1]:
            x = layer(x)
        return x

    def forward_with_features(self, x):
        features = self.forward_features(x)
        return self.fc(features), features

    def forward(self, x):
        return self.forward_with_features(x)[0]


class VGG11(_SemanticVGG):
    optimizer_kind = 'adam'

    def __init__(self, num_classes=10):
        super().__init__('cifar10', num_classes=num_classes)


class VGG16(_SemanticVGG):
    optimizer_kind = 'adam'

    def __init__(self, num_classes=10):
        super().__init__('cinic10', num_classes=num_classes)
