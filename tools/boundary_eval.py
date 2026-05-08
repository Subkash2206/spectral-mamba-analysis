import os
import sys
import glob
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import timm
from scipy.ndimage import binary_erosion

# Add current directory to path
sys.path.append(os.getcwd())
from models.vmunet.vmunet import VMUNet

class SwinSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True)
        self.head = nn.Conv2d(768, 1, 1)
        
    def forward(self, x):
        features = self.encoder(x)
        out = features[-1]
        
        # Format out to (B, C, H, W)
        if out.dim() == 3:
            B, L, C = out.shape
            H_f = W_f = int(math.sqrt(L))
            out = out.view(B, H_f, W_f, C).permute(0, 3, 1, 2)
        elif out.dim() == 4 and out.shape[-1] in [96, 192, 384, 768]:
            out = out.permute(0, 3, 1, 2)
            
        out = self.head(out)
        out = torch.nn.functional.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        return out

def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total

def compute_boundary_f1(pred, gt, iterations=2):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    # Erode
    pred_eroded = binary_erosion(pred, iterations=iterations)
    gt_eroded = binary_erosion(gt, iterations=iterations)
    
    # Extract boundary
    pred_bound = pred & ~pred_eroded
    gt_bound = gt & ~gt_eroded
    
    tp = (pred_bound & gt_bound).sum()
    fp = (pred_bound & ~gt_bound).sum()
    fn = (~pred_bound & gt_bound).sum()
    
    if tp + fp + fn == 0:
        return 1.0
    return 2.0 * tp / (2.0 * tp + fp + fn)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set seed for reproducible random Swin head
    torch.manual_seed(42)

    # 1. Load VM-UNet
    print('Loading VM-UNet...')
    vmunet = VMUNet().to(device)
    ckpt = torch.load('best-ckpt/best-vmunet-isic18.pth', map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        vmunet.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        vmunet.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        vmunet.load_state_dict(ckpt, strict=False)
    vmunet.eval()

    # 2. Load UNet-ResNet50
    print('Loading UNet-ResNet50...')
    unet = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
    unet.eval()

    # 3. Load Swin-Tiny
    print('Loading Swin-Tiny...')
    swin = SwinSeg().to(device)
    swin.eval()

    models = {
        'VM-UNet': vmunet,
        'UNet-ResNet50': unet,
        'Swin-Tiny': swin
    }

    # 4. Image paths
    img_dir = 'data/isic18/train/images/'
    mask_dir = 'data/isic18/train/masks/'
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))[:50]
    
    # Track metrics
    results = {
        'VM-UNet': {'dice': [], 'bf1': []},
        'UNet-ResNet50': {'dice': [], 'bf1': []},
        'Swin-Tiny': {'dice': [], 'bf1': []}
    }

    def get_transforms(size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    tfm_256 = get_transforms(256)
    tfm_224 = get_transforms(224)

    print(f'Running inference on {len(img_paths)} images...')
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]
        mask_name = base + '_segmentation.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        
        for m_name, model in models.items():
            size = 224 if m_name == 'Swin-Tiny' else 256
            tfm = tfm_224 if size == 224 else tfm_256
            
            x = tfm(img_pil).unsqueeze(0).to(device)
            # Resize mask to match
            gt = np.array(mask_pil.resize((size, size), Image.NEAREST))
            gt_bin = gt > 127
            
            with torch.no_grad():
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                
                # VM-UNet already applies sigmoid internally
                if m_name == 'VM-UNet':
                    pred = out.squeeze().cpu().numpy()
                else:
                    pred = torch.sigmoid(out).squeeze().cpu().numpy()
            
            pred_bin = pred > 0.5
            
            dice = compute_dice(pred_bin, gt_bin)
            bf1 = compute_boundary_f1(pred_bin, gt_bin, iterations=2)
            
            results[m_name]['dice'].append(dice)
            results[m_name]['bf1'].append(bf1)

    # 5. Print and save results
    out_csv = 'boundary_results.csv'
    print('\nBoundary Evaluation Results:')
    print('-' * 60)
    print(f'{"Model":<20} | {"Mean Dice":<15} | {"Mean BF1":<15}')
    print('-' * 60)
    
    with open(out_csv, 'w') as f:
        f.write('model,mean_dice,mean_bf1\n')
        for m_name in models.keys():
            mean_dice = np.mean(results[m_name]['dice'])
            mean_bf1 = np.mean(results[m_name]['bf1'])
            print(f'{m_name:<20} | {mean_dice:.4f}          | {mean_bf1:.4f}')
            f.write(f'{m_name},{mean_dice:.4f},{mean_bf1:.4f}\n')
    print('-' * 60)
    print(f'Saved results to {out_csv}')

if __name__ == '__main__':
    main()
