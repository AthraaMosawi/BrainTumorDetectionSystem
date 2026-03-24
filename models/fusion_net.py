import torch
import torch.nn as nn
import torchvision.models as models

class MultiEncoder(nn.Module):
    def __init__(self):
        super(MultiEncoder, self).__init__()
        # Use ResNet50 as backbone for each modality
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-2])) # Remove avgpool and fc
        
    def forward(self, x):
        return self.backbone(x)

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x_mri, x_xray, x_mwi):
        # Concatenate features
        combined = torch.cat([x_mri, x_xray, x_mwi], dim=1)
        # Calculate attention weights
        weights = self.attention(combined)
        
        # Weigh each modality
        fused = (weights[:, 0:1] * x_mri + 
                 weights[:, 1:2] * x_xray + 
                 weights[:, 2:3] * x_mwi)
        return fused

class TumorDetectionNet(nn.Module):
    def __init__(self):
        super(TumorDetectionNet, self).__init__()
        self.en_mri = MultiEncoder()
        self.en_xray = MultiEncoder()
        self.en_mwi = MultiEncoder()
        
        # Feature fusion (ResNet50 end features have 2048 channels)
        self.fusion = AttentionFusion(2048)
        
        # Segmentation Decoder (Simplified U-Net like)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, mri, xray, mwi):
        f_mri = self.en_mri(mri)
        f_xray = self.en_xray(xray)
        f_mwi = self.en_mwi(mwi)
        
        fused = self.fusion(f_mri, f_xray, f_mwi)
        
        seg = self.decoder(fused)
        cls = self.classifier(fused)
        
        return seg, cls

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
