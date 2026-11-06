"""DenseNet-121 pretrained on chest X-rays (TorchXRayVision), with a new single-output T2D head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet121XRV(nn.Module):
    def __init__(self, weights="densenet121-res224-all", dropout=0.3):
        super().__init__()
        import torchxrayvision as xrv
        net = xrv.models.DenseNet(weights=weights)       # DenseNet-121 + chest X-ray pretrained weights
        self.features = net.features                      # keep the pretrained feature layers
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1024, 1))  # new T2D output (logit)

    def forward(self, x):
        x = x * 2048.0 - 1024.0                           # [0,1] -> [-1024,1024], the range XRV expects
        f = F.relu(self.features(x), inplace=True)
        return self.head(torch.flatten(F.adaptive_avg_pool2d(f, 1), 1))

    def param_groups(self, lr, backbone_lr_mult=1.0):
        """Head trains at lr; pretrained layers at lr * backbone_lr_mult."""
        head_ids = {id(p) for p in self.head.parameters()}
        backbone = [p for p in self.parameters() if id(p) not in head_ids]
        return [{"params": list(self.head.parameters()), "lr": lr},
                {"params": backbone, "lr": lr * backbone_lr_mult}]


REGISTRY = {"densenet121_xrv": DenseNet121XRV}


def build_model(cfg):
    m = dict(cfg["model"])
    return REGISTRY[m.pop("name")](**m)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
