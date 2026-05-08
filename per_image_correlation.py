import sys
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

# Add paths for models
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
        self.opts = None; self.batch_size = 1; self.zip = False; self.cache_mode = 'part'
        self.resume = None; self.accumulation_steps = None; self.use_checkpoint = False
        self.amp_opt_level = 'O0'; self.tag = 'test'; self.eval = False; self.throughput = False

def compute_avr(fmap):
    fmap = fmap.cpu().float()
    fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap); fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1)); power = torch.abs(fft_shifted) ** 2
    cy, cx = H // 2, W // 2; y = torch.arange(H).view(1, 1, H, 1); x = torch.arange(W).view(1, 1, 1, W)
    mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4); mask = mask.expand(B, C, H, W)
    return (power * mask).sum().item() / power.sum().item() if power.sum() > 0 else 0.0

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
    if tp + fp + fn == 0:
        return 1.0
    return 2.0 * tp / (2.0 * tp + fp + fn)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running Per-Image Correlation on {device}...')

    ckpt_dir = os.path.join(ROOT, 'VM-UNet/best-ckpt/')
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print("Loading models...")
    unet = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device)
    unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-unet-isic18.pth'), map_location=device)); unet.eval()
    
    args = MockArgs(); config = get_config(args); swin = SwinUnet(config, img_size=224, num_classes=1).to(device)
    swin.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-swinunet-isic18.pth'), map_location=device)); swin.eval()
    
    vmunet = VMUNet().to(device)
    vmunet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth'), map_location=device), strict=True)
    vmunet.eval()

    features = defaultdict(dict)
    def get_hook(model_name, level):
        def hook(module, input, output):
            if isinstance(output, tuple): features[model_name][level] = output[0].detach()
            else: features[model_name][level] = output.detach()
        return hook

    for i in range(1, 5):
        getattr(unet.encoder, f'layer{i}').register_forward_hook(get_hook('UNet', i))
        swin.swin_unet.layers[i-1].blocks[-1].register_forward_hook(get_hook('Swin', i))
        vmunet.vmunet.layers[i-1].blocks[-1].register_forward_hook(get_hook('Mamba', i))

    img_dir = os.path.join(ROOT, 'VM-UNet/data/isic18/train/images/')
    mask_dir = os.path.join(ROOT, 'VM-UNet/data/isic18/train/masks/')
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    import random; random.seed(42); random.shuffle(img_paths)
    split_idx = int(0.8 * len(img_paths))
    val_imgs = img_paths[split_idx:split_idx+50] # 50 images for correlation is standard

    results = {'UNet': {'avr': [], 'bf1': []}, 'Swin': {'avr': [], 'bf1': []}, 'Mamba': {'avr': [], 'bf1': []}}

    print(f'Auditing {len(val_imgs)} images...')
    for idx, img_path in enumerate(val_imgs):
        if idx % 10 == 0: print(f"Processing image {idx}/{len(val_imgs)}...")
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, base + '_segmentation.png')
        
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        
        i256 = t256(img_pil).unsqueeze(0).to(device)
        i224 = t224(img_pil).unsqueeze(0).to(device)
        
        gt_256 = np.array(mask_pil.resize((256, 256), Image.NEAREST)) > 127
        gt_224 = np.array(mask_pil.resize((224, 224), Image.NEAREST)) > 127

        with torch.no_grad():
            features.clear()
            out_unet = unet(i256)
            out_swin = swin(i224)
            out_mamba = vmunet(i256)
            
            pred_unet = (torch.sigmoid(out_unet).squeeze().cpu().numpy() > 0.5)
            pred_swin = (torch.sigmoid(out_swin).squeeze().cpu().numpy() > 0.5)
            # VM-UNet already applies sigmoid
            pred_mamba = (out_mamba.squeeze().cpu().numpy() > 0.5)
            
            results['UNet']['bf1'].append(compute_boundary_f1(pred_unet, gt_256))
            results['Swin']['bf1'].append(compute_boundary_f1(pred_swin, gt_224))
            results['Mamba']['bf1'].append(compute_boundary_f1(pred_mamba, gt_256))

            # Compute AVR
            for model_name in ['UNet', 'Swin', 'Mamba']:
                avrs = []
                for i in range(1, 5):
                    f = features[model_name][i]
                    if model_name == 'Swin':
                        if f.dim() == 3:
                            B, L, C = f.shape; H = W = int(np.sqrt(L)); f = f.transpose(1, 2).reshape(B, C, H, W)
                        elif f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]:
                            f = f.permute(0, 3, 1, 2)
                    elif model_name == 'Mamba':
                        if f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]:
                            f = f.permute(0, 3, 1, 2)
                    avrs.append(compute_avr(f))
                results[model_name]['avr'].append(np.mean(avrs))

    print('\n' + '='*60)
    print("Correlation Results (Mean AVR vs BF1)")
    print('-'*60)
    
    csv_data = []
    all_avr = []
    all_bf1 = []
    
    for model_name in ['UNet', 'Swin', 'Mamba']:
        avr_arr = np.array(results[model_name]['avr'])
        bf1_arr = np.array(results[model_name]['bf1'])
        r, p = pearsonr(avr_arr, bf1_arr)
        print(f'{model_name:<15} | r = {r:7.4f} | p = {p:.4e}')
        csv_data.append(f"{model_name},{r},{p}")
        
        all_avr.extend(avr_arr)
        all_bf1.extend(bf1_arr)
        
    r_all, p_all = pearsonr(all_avr, all_bf1)
    print('-'*60)
    print(f'{"Pooled (n=150)":<15} | r = {r_all:7.4f} | p = {p_all:.4e}')
    csv_data.append(f"Pooled,{r_all},{p_all}")
    
    # Partial correlation controlling for model identity
    all_avr_arr = np.array(all_avr)
    all_bf1_arr = np.array(all_bf1)
    
    # Create one-hot matrix for the 3 models (50 each)
    X = np.zeros((150, 3))
    X[0:50, 0] = 1
    X[50:100, 1] = 1
    X[100:150, 2] = 1
    
    # Regress AVR on model identity
    beta_avr, _, _, _ = np.linalg.lstsq(X, all_avr_arr, rcond=None)
    resid_avr = all_avr_arr - X.dot(beta_avr)
    
    # Regress BF1 on model identity
    beta_bf1, _, _, _ = np.linalg.lstsq(X, all_bf1_arr, rcond=None)
    resid_bf1 = all_bf1_arr - X.dot(beta_bf1)
    
    # Compute Pearson r between residuals
    r_part, p_part = pearsonr(resid_avr, resid_bf1)
    
    print(f'{"Partial (n=150)":<15} | r = {r_part:7.4f} | p = {p_part:.4e}')
    print('='*60)
    csv_data.append(f"Partial,{r_part},{p_part}")
    
    with open('correlation_results.csv', 'w') as f:
        f.write('model,pearson_r,p_value\n')
        f.write('\n'.join(csv_data) + '\n')
    print('Saved results to correlation_results.csv')

if __name__ == '__main__':
    main()
