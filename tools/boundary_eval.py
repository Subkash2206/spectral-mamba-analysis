import os
import sys
import glob
import math
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion
from tqdm import tqdm

# Add project root to path
ROOT = os.getcwd()
sys.path.append(ROOT)
from models.vmunet.vmunet import VMUNet

# Import SwinUnet
sys.path.append(os.path.join(ROOT, 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = os.path.join(ROOT, 'Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml')
        self.opts = None; self.batch_size = 1; self.zip = False; self.cache_mode = 'part'; self.resume = None; self.accumulation_steps = None; self.use_checkpoint = False; self.amp_opt_level = 'O0'; self.tag = 'test'; self.eval = False; self.throughput = False

def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0: return 1.0
    return 2.0 * intersection / total

def compute_boundary_f1(pred, gt, iterations=2):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_eroded = binary_erosion(pred, iterations=iterations)
    gt_eroded = binary_erosion(gt, iterations=iterations)
    pred_bound = pred & ~pred_eroded
    gt_bound = gt & ~gt_eroded
    tp = (pred_bound & gt_bound).sum()
    fp = (pred_bound & ~gt_bound).sum()
    fn = (~pred_bound & gt_bound).sum()
    if tp + fp + fn == 0: return 1.0
    return 2.0 * tp / (2.0 * tp + fp + fn)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'UTMOST PRECISION AUDIT: Starting Full Validation Evaluation on {device}...')

    # 1. Reproduce Training Split (Seed 42)
    img_dir = os.path.join(ROOT, 'VM-UNet/data/isic18/train/images/')
    mask_dir = os.path.join(ROOT, 'VM-UNet/data/isic18/train/masks/')
    all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    random.seed(42)
    random.shuffle(all_imgs)
    split_idx = int(0.8 * len(all_imgs))
    val_imgs = all_imgs[split_idx:]
    print(f'Total Images: {len(all_imgs)} | Full Validation Set Size: {len(val_imgs)}')

    # 2. Load TRAINED Models
    ckpt_dir = os.path.join(ROOT, 'VM-UNet/best-ckpt/')
    
    print('Loading VM-UNet (Mamba)...')
    vmunet = VMUNet().to(device)
    vmunet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth'), map_location=device))
    vmunet.eval()

    print('Loading UNet-ResNet50...')
    unet = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device)
    unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-unet-isic18.pth'), map_location=device))
    unet.eval()

    print('Loading Swin-Tiny...')
    args = MockArgs()
    config = get_config(args)
    swin = SwinUnet(config, img_size=224, num_classes=1).to(device)
    swin.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-swinunet-isic18.pth'), map_location=device))
    swin.eval()

    models = {'VM-UNet': vmunet, 'UNet-ResNet50': unet, 'Swin-Tiny': swin}

    results = {name: {'dice': [], 'bf1': []} for name in models}
    
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for img_path in tqdm(val_imgs, desc='Evaluating'):
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = base + '_segmentation.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path): continue
            
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        
        for m_name, model in models.items():
            size = 224 if m_name == 'Swin-Tiny' else 256
            tfm = t224 if size == 224 else t256
            x = tfm(img_pil).unsqueeze(0).to(device)
            gt = np.array(mask_pil.resize((size, size), Image.NEAREST)) > 127
            
            with torch.no_grad():
                out = model(x)
                if isinstance(out, tuple): out = out[0]
                pred = out.squeeze().cpu().numpy() if m_name == 'VM-UNet' else torch.sigmoid(out).squeeze().cpu().numpy()
            
            pred_bin = pred > 0.5
            results[m_name]['dice'].append(compute_dice(pred_bin, gt))
            results[m_name]['bf1'].append(compute_boundary_f1(pred_bin, gt))

    # Save results
    out_csv = os.path.join(ROOT, 'boundary_results.csv')
    with open(out_csv, 'w') as f:
        f.write('model,mean_dice,mean_bf1\n')
        for m_name in models:
            d = np.mean(results[m_name]['dice'])
            b = np.mean(results[m_name]['bf1'])
            f.write(f'{m_name},{d:.4f},{b:.4f}\n')
            print(f'{m_name:<15} | Dice: {d:.4f} | BF1: {b:.4f}')

if __name__ == '__main__':
    main()
