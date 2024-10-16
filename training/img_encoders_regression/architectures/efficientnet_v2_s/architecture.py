import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class EfficientNetV2SModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet_v2_s = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.fc = nn.Linear(self.efficientnet_v2_s.classifier[1].out_features, 4)

    def forward(self, x):
        x = self.efficientnet_v2_s(x)
        x = self.fc(x)
        return x

    @staticmethod
    def transforms_for_model():
        return transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
