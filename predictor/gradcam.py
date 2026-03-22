import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2

# Inference configuration 
THRESHOLD   = 0.5  
TTA_ENABLED = False
INPUT_SIZE  = 224    
_CACHED_TARGET_LAYER = None


def _get_target_layer(model):
    backbone   = model.backbone
    candidates = [
        ("act2",      "After BN + SiLU — sharpest CAMs"),
        ("bn2",       "After BatchNorm — good CAMs"),
        ("conv_head", "Pre-activation — fallback"),
    ]
    for attr, desc in candidates:
        if hasattr(backbone, attr):
            layer = getattr(backbone, attr)
            print(f"[GradCAM] Target layer: backbone.{attr} ({desc})")
            return layer

    print("[GradCAM] WARNING: Known layers not found. Using last Conv2d.")
    last_conv = None
    for _, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise RuntimeError(
        "[GradCAM] No suitable target layer found.\n"
        "Run: print([n for n, _ in model.backbone.named_children()])"
    )


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def _tta_views(image_tensor):
    """5 TTA views: original, flip, bright+, bright-, flip+bright."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3,1,1)
    img  = (image_tensor * std + mean).clamp(0, 1)

    def norm(t):
        return (t.clamp(0, 1) - mean) / std

    return torch.stack([
        image_tensor,
        norm(TF.hflip(img)),
        norm((img * 1.10).clamp(0, 1)),
        norm((img * 0.90).clamp(0, 1)),
        norm((TF.hflip(img) * 1.10).clamp(0, 1))
    ])


def _tta_probability(model, rgb_tensor):
    views = _tta_views(rgb_tensor.squeeze(0))
    with torch.no_grad():
        probs = torch.sigmoid(model(views)).squeeze(1)
    return probs.mean().item()


class GradCAM:
    
    def __init__(self, model, target_layer):
        self.model          = model
        self.target_layer   = target_layer
        self.gradients      = None
        self.activations    = None
        self._forward_count = 0
        self._hooks         = []
        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            if self._forward_count == 0:
                self.activations = output.detach()
            self._forward_count += 1

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, rgb_tensor):
        self.gradients      = None
        self.activations    = None
        self._forward_count = 0

        self.model.zero_grad()

        with torch.enable_grad():
            rgb_tensor = rgb_tensor.detach().requires_grad_(True)
            output     = self.model(rgb_tensor)
            score      = torch.sigmoid(output)
            score.backward()

        if self.activations is None:
            self.remove_hooks()
            raise RuntimeError("GradCAM forward hook did not fire.")
        if self.gradients is None:
            self.remove_hooks()
            raise RuntimeError("GradCAM backward hook did not fire.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam     = F.relu(torch.sum(weights * self.activations, dim=1))

        # v5: upsample to 224×224 (was 128×128)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        self.remove_hooks()
        return cam, score.item(), rgb_tensor


def overlay_heatmap(original_image, cam, alpha=0.3):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        original_image.astype(np.uint8),
        0.7,
        heatmap.astype(np.uint8),
        0.3,
        0
    )
    return overlay

def predict_with_gradcam(model, rgb_tensor, threshold=None):    
    global _CACHED_TARGET_LAYER

    if threshold is None:
        threshold = THRESHOLD

    device     = next(model.parameters()).device
    rgb_tensor = rgb_tensor.to(device)

    # ── Probability ───────────────────────────────────────────────────────
    if TTA_ENABLED:
        probability = _tta_probability(model, rgb_tensor)
    else:
        with torch.no_grad():
            probability = torch.sigmoid(model(rgb_tensor)).item()

    prediction = 1 if probability > threshold else 0
    
    if _CACHED_TARGET_LAYER is None:
        _CACHED_TARGET_LAYER = _get_target_layer(model)

    gradcam_obj = GradCAM(model, _CACHED_TARGET_LAYER)
    cam, _, rgb_tensor = gradcam_obj.generate(rgb_tensor)

    # ── Reconstruct original image at 224×224 ─────────────────────────────
    rgb_denorm = denormalize(rgb_tensor.squeeze(0).detach())
    original   = (rgb_denorm.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    overlay    = overlay_heatmap(original, cam)

    return prediction, probability, overlay, cam