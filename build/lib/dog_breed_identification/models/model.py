import torch
import timm


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('mobilenetv3_large_100',
                                       pretrained=True, num_classes=120)

    def forward(self, x):
        return self.model(x)
