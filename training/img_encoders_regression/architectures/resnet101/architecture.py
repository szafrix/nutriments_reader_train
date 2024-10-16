import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class ResNet101Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet101 = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2
        )
        self.fc = nn.Linear(self.resnet101.fc.out_features, 4)

    def forward(self, x):
        x = self.resnet101(x)
        x = self.fc(x)
        return x

    @staticmethod
    def transforms_for_model():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                # lambda x: x.unsqueeze(0),
            ]
        )
