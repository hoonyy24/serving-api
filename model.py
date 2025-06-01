# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from vit_pytorch import ViT

class Classifier(nn.Module):
    def __init__(self, num_classes=14):
        super(Classifier, self).__init__()
        backbone = resnet50(pretrained=False) 
        self.conv1 = backbone.conv1
        self.bn1   = backbone.bn1
        self.relu  = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1   = backbone.layer1
        self.layer2   = backbone.layer2
        self.layer3   = backbone.layer3
        self.layer4   = backbone.layer4

        self.vit_pre_conv = nn.Conv2d(1024, 3, kernel_size=1)
        self.vit_branch = ViT(
            image_size=16,
            patch_size=4,
            num_classes=1024,
            dim=512,
            depth=10,
            heads=12,
            mlp_dim=1024,
            dropout=0.2,
            emb_dropout=0.2
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.aux_classifier = nn.Linear(512, num_classes)

        self.cnn_proj = nn.Linear(2048, 256)
        self.vit_proj = nn.Linear(1024, 256)

        self.attention = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        aux_feature = self.pool(f2)
        aux_feature = aux_feature.view(f2.size(0), -1)
        aux_out = self.aux_classifier(aux_feature)

        vit_input = F.interpolate(f3, size=(16, 16), mode='bilinear', align_corners=False)
        vit_input = self.vit_pre_conv(vit_input)
        vit_features = self.vit_branch(vit_input)

        cnn_feature = self.pool(f4)
        cnn_feature = cnn_feature.view(f4.size(0), -1)

        cnn_proj = self.cnn_proj(cnn_feature)
        vit_proj = self.vit_proj(vit_features)

        attn_input = torch.cat([cnn_proj, vit_proj], dim=1)
        attn_weights = self.attention(attn_input)
        weight_cnn = attn_weights[:, 0].unsqueeze(1)
        weight_vit = attn_weights[:, 1].unsqueeze(1)

        attended_feature = weight_cnn * cnn_proj + weight_vit * vit_proj
        bilinear_feature = cnn_proj * vit_proj

        fusion = torch.cat([attended_feature, bilinear_feature], dim=1)
        out = self.fc(fusion)

        return out if not self.training else (out, aux_out)
