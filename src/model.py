"""
EfficientNetV2-S model factory with freeze/unfreeze utilities for 2-phase fine-tuning.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, Swin_T_Weights, Swin_V2_T_Weights

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def create_model(num_classes=config.NUM_CLASSES, pretrained=True):
    """
    Create a model (EfficientNetV2-S, ConvNeXt-Tiny, or Swin-T) based on config.
    """
    arch = getattr(config, "MODEL_ARCH", "efficientnet_v2_s")
    
    if arch == "swin_v2_t":
        if pretrained:
            weights = Swin_V2_T_Weights.IMAGENET1K_V1
            model = models.swin_v2_t(weights=weights)
            print("[Model] Loaded Swin-V2-T with IMAGENET1K_V1 pretrained weights")
        else:
            model = models.swin_v2_t(weights=None)
            print("[Model] Loaded Swin-V2-T without pretrained weights")
            
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
    elif arch == "swin_t":
        if pretrained:
            weights = Swin_T_Weights.IMAGENET1K_V1
            model = models.swin_t(weights=weights)
            print("[Model] Loaded Swin-T with IMAGENET1K_V1 pretrained weights")
        else:
            model = models.swin_t(weights=None)
            print("[Model] Loaded Swin-T without pretrained weights")
            
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
    elif arch == "convnext_tiny":
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            model = models.convnext_tiny(weights=weights)
            print("[Model] Loaded ConvNeXt-Tiny with IMAGENET1K_V1 pretrained weights")
        else:
            model = models.convnext_tiny(weights=None)
            print("[Model] Loaded ConvNeXt-Tiny without pretrained weights")
            
        # ConvNeXt classifier is: Sequential(LayerNorm, Flatten, Linear)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
    elif arch == "convnext_small":
        if pretrained:
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
            model = models.convnext_small(weights=weights)
            print("[Model] Loaded ConvNeXt-Small with IMAGENET1K_V1 pretrained weights")
        else:
            model = models.convnext_small(weights=None)
            print("[Model] Loaded ConvNeXt-Small without pretrained weights")
            
        # ConvNeXt classifier is: Sequential(LayerNorm, Flatten, Linear)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
    else:
        # Default to EfficientNetV2-S
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            model = models.efficientnet_v2_s(weights=weights)
            print("[Model] Loaded EfficientNetV2-S with IMAGENET1K_V1 pretrained weights")
        else:
            model = models.efficientnet_v2_s(weights=None)
            print("[Model] Loaded EfficientNetV2-S without pretrained weights")

        # EfficientNetV2 classifier is: Sequential(Dropout, Linear)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    print(f"[Model] Replaced classifier head: {in_features} -> 512 -> {num_classes}")
    return model


def freeze_backbone(model):
    """
    Freeze all layers except the final classification head.
    Used during Phase 1 (warmup) training.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if "classifier" not in name and "head" not in name:
            param.requires_grad = False
            frozen += 1
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] Backbone FROZEN: {frozen} param groups frozen")
    print(f"[Model] Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")


def unfreeze_backbone(model):
    """
    Unfreeze all layers for Phase 2 fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] Backbone UNFROZEN: all {total:,} params are trainable")


def get_parameter_groups(model, backbone_lr, head_lr):
    """
    Separate parameters into backbone and head for differential learning rates.
    """
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    print(f"[Model] Parameter groups: backbone_lr={backbone_lr}, head_lr={head_lr}")
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
