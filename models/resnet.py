import torch.nn as nn
import torchvision.models as models

def get_resnet_classifier(num_classes=2):
    """
    Carica ResNet-18 pre-addestrata su ImageNet e modifica 
    l'ultimo layer fully-connected per il numero specificato di classi.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Sostituiamo l'ultimo strato (Fully Connected)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
