import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FrequencyBranch(nn.Module):
   
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1)
        fft  = torch.fft.fftshift(torch.fft.fft2(gray))
        mag  = torch.log(torch.abs(fft) + 1e-8)
        h, w  = mag.shape[-2], mag.shape[-1]
        hf_region = mag[:, int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        hf_energy = hf_region.mean().item()
        print(f"[FFT DIAGNOSTIC] HF energy: {hf_energy:.4f}")
        mag  = self.pool(mag.unsqueeze(1)).squeeze(1)
        mag  = mag.flatten(1)
        return self.proj(mag)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class HybridModel(nn.Module):

    PERTURBATION_KERNEL = 5

    def __init__(self, pretrained: bool = False, se_reduction: int = 16):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = ""
        )
        self.pool        = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1536
        self.freq_dim    = 256
        self.fusion_dim  = self.feature_dim * 2 + self.freq_dim  

        self.freq_branch  = FrequencyBranch(out_dim=self.freq_dim)
        self.se_attention = SEBlock(self.fusion_dim, reduction=se_reduction)
        self.classifier   = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.backbone(x)), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_orig = self.extract_features(x)

        x_pert = F.avg_pool2d(
            x,
            kernel_size = self.PERTURBATION_KERNEL,
            stride      = 1,
            padding     = self.PERTURBATION_KERNEL // 2
        )
        f_sub  = f_orig - self.extract_features(x_pert)
        f_freq = self.freq_branch(x)

        fusion = torch.cat([f_orig, f_sub, f_freq], dim=1)  
        fusion = self.se_attention(fusion)
        return self.classifier(fusion)