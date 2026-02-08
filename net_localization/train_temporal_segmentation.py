"""
Temporal segmentation training for sonar net detection.

Uses ConvLSTM + U-Net architecture to leverage temporal information from video sequences.

Requirements:
    pip install torch torchvision segmentation-models-pytorch albumentations

Usage Examples:
    # Quick test with minimal data (VERY FAST - for testing/debugging)
    python train_temporal_segmentation.py --data_dirs ../data/runs/temp_augmentation_* --epochs 10 --max_frames_per_run 30 --sequence_length 3 --batch_size 8
    
    # Preview resolution debugging (test different image sizes without training)
    python train_temporal_segmentation.py --data_dirs ../data/runs/temp_augmentation_* --preview_image_size 100 --skip_preview
    python train_temporal_segmentation.py --data_dirs ../data/runs/temp_augmentation_* --preview_image_size 128 --debug_preview --mask_threshold 5.0
    
    # Zoomed training (10m range with 2x spatial detail)
    python train_temporal_segmentation.py --data_dirs ../data/runs/temp_augmentation_* --max_range 10.0 --original_range 20.0 --epochs 50 --batch_size 8
    
    # Fast training with signal-filtered ground truth (recommended)
    python train_temporal_segmentation.py --data_dirs ../data/runs/temp_augmentation_* --epochs 50 --sequence_length 3 --image_size 256 --batch_size 8 --mask_threshold 10.0
    
    # Full fast training with subset of data (good for iteration)
    python train_temporal_segmentation.py --data_dirs ../data/runs/temp_augmentation_* --epochs 50 --max_frames_per_run 100 --sequence_length 3 --image_size 256 --batch_size 8
    
    # High quality training (SLOWER but better results)
    python train_temporal_segmentation.py --data_dirs ../data/runs/temp_augmentation_* --epochs 50 --sequence_length 5 --image_size 320 --batch_size 4 --mask_threshold 10.0
    
    # Maximum quality (VERY SLOW)
    python train_temporal_segmentation.py --data_dirs ../data/runs/temp_augmentation_* --epochs 100 --sequence_length 5 --image_size 400 --batch_size 2 --encoder resnet50 --mask_threshold 10.0

Performance Tuning Guide:
    Speed (fastest to slowest):
        --image_size: 256 > 320 > 400 (2.4x faster: 256 vs 400)
        --sequence_length: 3 > 5 > 7 (1.7x faster: 3 vs 5)
        --batch_size: 16 > 8 > 4 > 2 (better GPU utilization)
        --encoder: resnet18 > resnet34 > resnet50 (1.5x faster: resnet18 vs resnet34)
        --max_frames_per_run: 50 > 100 > None (proportional speedup)
        --num_workers: 4 > 2 > 0 (2-3x faster data loading: 2 vs 0)
        --preview_image_size: Use lower resolution (100-128) for quick preview checks
    
    Quality (from most important to least):
        --sequence_length: 5 better than 3 (more temporal context)
        --image_size: 320-400 better than 256 (more spatial detail)
        --encoder: resnet50 > resnet34 > resnet18 (better features)
        --epochs: 50-100 (more training time)
        --batch_size: larger = smoother gradients (but diminishing returns)
        --mask_threshold: 5-10 recommended (filters impossible ground truth in shadowed areas)
        --max_range: Use 10m for 2x zoom on close objects, 20m for full view
    
    Ground Truth Filtering:
        --mask_threshold 0: No filtering (trains on all GT, including dark/shadowed areas)
        --mask_threshold 1-5: Light filtering (keeps most visible signals)
        --mask_threshold 10: Recommended (removes GT in low-signal areas)
        --mask_threshold 20+: Aggressive filtering (only strong returns)
        Use --debug_preview to visualize before/after filtering
    
    Memory Usage (if you get OOM errors):
        Reduce --batch_size: try 4 → 2 → 1
        Reduce --image_size: try 320 → 256 → 224
        Reduce --sequence_length: try 5 → 3
        Use --accumulation_steps 2 to simulate larger batches
        Switch to smaller encoder: resnet34 → resnet18
    
    Recommended Presets:
        Quick test:       --max_sequences 100 --epochs 10 --batch_size 8 --preview_image_size 128
        Development:      --max_frames_per_run 100 --epochs 30 --batch_size 8 --mask_threshold 10
        Production:       --sequence_length 3 --image_size 256 --batch_size 8 --epochs 50 --mask_threshold 10
        Best quality:     --sequence_length 5 --image_size 320 --epochs 100 --encoder resnet50 --mask_threshold 10
        Zoomed training:  --max_range 10.0 --original_range 20.0 --image_size 256 --mask_threshold 5
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import glob
import cv2
import matplotlib.pyplot as plt


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for temporal modeling"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(self, x, hidden_state):
        h, c = hidden_state
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class TemporalUNet(nn.Module):
    """U-Net with ConvLSTM for temporal segmentation"""
    def __init__(self, encoder_name='resnet34', sequence_length=5, hidden_dim=64):
        super().__init__()
        self.sequence_length = sequence_length
        
        # Spatial encoder (shared across time)
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=hidden_dim,
            activation=None
        )
        
        # Temporal modeling with ConvLSTM
        self.convlstm = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)
        
        # Final prediction head
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch, time, channels, height, width
        Returns:
            (B, 1, H, W) - segmentation for the last frame
        """
        batch_size, seq_len, C, H, W = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.convlstm.hidden_dim, H, W, device=x.device)
        c = torch.zeros(batch_size, self.convlstm.hidden_dim, H, W, device=x.device)
        
        # Process sequence
        for t in range(seq_len):
            # Spatial features for frame t
            spatial_features = self.encoder(x[:, t])  # (B, hidden_dim, H, W)
            
            # Temporal update
            h, (h, c) = self.convlstm(spatial_features, (h, c))
        
        # Predict segmentation for last frame
        out = self.final_conv(h)
        
        return out


class SonarSequenceDataset(Dataset):
    """Load sequences of sonar frames from simulation NPZ files"""
    def __init__(self, data_dirs, sequence_length=5, stride=2, transform=None, max_range=20.0, fov=120.0, image_size=256, max_frames_per_run=None, max_sequences=None, original_range_m=20.0, mask_threshold=10.0, max_frames_list=None):
        """
        Args:
            data_dirs: List of paths to simulation run directories
            sequence_length: Number of frames per sequence
            stride: Stride between sequences (stride=1 means overlapping sequences)
            transform: Albumentations transforms
            max_range: Sonar max range to display in meters (for zooming/cropping)
            fov: Sonar field of view in degrees
            image_size: Output image size (default 256 for faster training)
            max_frames_per_run: Maximum frames to load from each run (None = all, used as fallback if max_frames_list not provided)
            max_sequences: Maximum total sequences (None = all)
            original_range_m: Original range the polar data was collected at (default 20.0m)
            mask_threshold: Minimum sonar intensity for pixel to be marked as net (0 = no filtering)
            max_frames_list: List of frame limits for each data_dir (overrides max_frames_per_run for specific dirs)
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.max_range = max_range
        self.original_range_m = original_range_m
        self.fov = fov
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        
        # Collect all frame files from all directories
        self.sequences = []
        for dir_idx, data_dir in enumerate(data_dirs):
            data_path = Path(data_dir)
            frames_dir = data_path / 'frames'
            
            if not frames_dir.exists():
                print(f"Warning: {frames_dir} does not exist, skipping")
                continue
            
            frame_files = sorted(frames_dir.glob('frame_*.npz'))
            
            # Determine frame limit for this directory
            if max_frames_list and dir_idx < len(max_frames_list):
                frame_limit = max_frames_list[dir_idx]
            else:
                frame_limit = max_frames_per_run
            
            # Limit frames per run for faster training
            if frame_limit:
                frame_files = frame_files[:frame_limit]
                print(f"  {data_path.name}: Loading {len(frame_files)} frames (limit: {frame_limit})")
            else:
                print(f"  {data_path.name}: Loading {len(frame_files)} frames (no limit)")
            
            # Create sequences with stride
            for i in range(0, len(frame_files) - sequence_length + 1, stride):
                seq = frame_files[i:i + sequence_length]
                if len(seq) == sequence_length:
                    self.sequences.append(seq)
        
        # Limit total sequences
        if max_sequences and len(self.sequences) > max_sequences:
            self.sequences = self.sequences[:max_sequences]
        
        print(f"Created {len(self.sequences)} sequences from {len(data_dirs)} runs")
    
    def polar_to_cartesian(self, polar_image, output_size=None, is_mask=False):
        """Convert polar sonar image to Cartesian coordinates with proper zoom support.
        
        When max_range < original_range_m, this 'zooms in' by only sampling
        the near-field portion of the polar data.
        
        Args:
            polar_image: Polar format image (r_bins, n_beams)
            output_size: Output image size
            is_mask: If True, use nearest-neighbor interpolation for discrete data
        """
        if output_size is None:
            output_size = self.image_size
        r_bins, n_beams = polar_image.shape
        half_width = self.max_range * np.sin(np.radians(self.fov / 2))
        x = np.linspace(-half_width, half_width, output_size)
        y = np.linspace(0, self.max_range, output_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(X, Y)
        fov_rad = np.radians(self.fov)
        beam_idx = (Theta + fov_rad / 2) / fov_rad * (n_beams - 1)
        # CRITICAL: Map to original range, not display range (enables zoom)
        range_idx = (R / self.original_range_m) * (r_bins - 1)
        beam_idx = np.clip(beam_idx, 0, n_beams - 1).astype(np.float32)
        range_idx = np.clip(range_idx, 0, r_bins - 1).astype(np.float32)
        
        # Use nearest neighbor for masks (discrete data), linear for sonar intensity
        interp_method = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        cart = cv2.remap(polar_image.astype(np.float32), beam_idx, range_idx, 
                         interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Mask out regions outside display range
        cart[(R > self.max_range) | (Theta < -fov_rad/2) | (Theta > fov_rad/2)] = 0
        return cart
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_files = self.sequences[idx]
        
        images = []
        masks = []
        
        for frame_path in sequence_files:
            data = np.load(frame_path)
            sonar_img = data['sonar_image']
            ground_truth = data['ground_truth']
            data.close()
            
            # Convert to Cartesian
            img_cart = self.polar_to_cartesian(sonar_img, is_mask=False)
            mask_polar = (ground_truth == 1).astype(np.float32)  # Net material ID = 1
            mask_cart = self.polar_to_cartesian(mask_polar, is_mask=True)  # Use nearest-neighbor for masks
            mask_cart = (mask_cart > 0.5).astype(np.float32)
            
            # Filter mask to only include pixels with sufficient signal strength
            if self.mask_threshold > 0:
                mask_cart = mask_cart * (img_cart > self.mask_threshold).astype(np.float32)
            
            # Normalize image to 0-255
            img_norm = ((img_cart - img_cart.min()) / 
                       (img_cart.max() - img_cart.min() + 1e-8) * 255).astype(np.uint8)
            
            # Convert to 3-channel
            img_3ch = np.stack([img_norm, img_norm, img_norm], axis=-1)
            
            images.append(img_3ch)
            masks.append(mask_cart)
        
        # Apply same transform to all frames in sequence
        if self.transform:
            # Transform each frame
            transformed_images = []
            transformed_masks = []
            for img, mask in zip(images, masks):
                transformed = self.transform(image=img, mask=mask)
                transformed_images.append(transformed['image'])
                transformed_masks.append(transformed['mask'])
            
            images = torch.stack(transformed_images)  # (T, C, H, W)
            masks = torch.stack(transformed_masks)  # (T, H, W)
        
        # Return last mask as target
        return images, masks[-1].unsqueeze(0)  # (T, C, H, W), (1, H, W)


class PreSplitSonarDataset(Dataset):
    """Dataset for pre-split sequences (used for per-directory 80/20 splits)"""
    def __init__(self, sequences, parent_dataset):
        self.sequences = sequences
        self.parent = parent_dataset
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_files = self.sequences[idx]
        images = []
        masks = []
        
        for frame_path in sequence_files:
            data = np.load(frame_path)
            sonar_img = data['sonar_image']
            ground_truth = data['ground_truth']
            data.close()
            
            img_cart = self.parent.polar_to_cartesian(sonar_img, is_mask=False)
            mask_polar = (ground_truth == 1).astype(np.float32)
            mask_cart = self.parent.polar_to_cartesian(mask_polar, is_mask=True)
            mask_cart = (mask_cart > 0.5).astype(np.float32)
            
            if self.parent.mask_threshold > 0:
                mask_cart = mask_cart * (img_cart > self.parent.mask_threshold).astype(np.float32)
            
            img_norm = ((img_cart - img_cart.min()) / 
                       (img_cart.max() - img_cart.min() + 1e-8) * 255).astype(np.uint8)
            img_3ch = np.stack([img_norm, img_norm, img_norm], axis=-1)
            
            images.append(img_3ch)
            masks.append(mask_cart)
        
        if self.parent.transform:
            transformed_images = []
            transformed_masks = []
            for img, mask in zip(images, masks):
                transformed = self.parent.transform(image=img, mask=mask)
                transformed_images.append(transformed['image'])
                transformed_masks.append(transformed['mask'])
            
            images = torch.stack(transformed_images)
            masks = torch.stack(transformed_masks)
        
        return images, masks[-1].unsqueeze(0)


class AugmentedSequenceDataset(Dataset):
    """Load sequences from augmented PNG training data"""
    def __init__(self, data_dir, split='train', sequence_length=5, stride=2, transform=None):
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        img_dir = Path(data_dir) / split / 'images'
        self.mask_dir = Path(data_dir) / split / 'masks'
        
        all_files = sorted(img_dir.glob('*.png'))
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(all_files) - sequence_length + 1, stride):
            self.sequences.append(all_files[i:i + sequence_length])
        
        print(f"{split}: {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_files = self.sequences[idx]
        
        images = []
        masks = []
        
        for img_path in sequence_files:
            mask_path = self.mask_dir / img_path.name
            
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            
            # Convert to 3-channel
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            
            mask = (mask > 127).astype(np.float32)
            
            images.append(image)
            masks.append(mask)
        
        # Apply transforms
        if self.transform:
            transformed_images = []
            transformed_masks = []
            for img, mask in zip(images, masks):
                transformed = self.transform(image=img, mask=mask)
                transformed_images.append(transformed['image'])
                transformed_masks.append(transformed['mask'])
            
            images = torch.stack(transformed_images)
            masks = torch.stack(transformed_masks)
        
        return images, masks[-1].unsqueeze(0)


def get_transforms(train=True):
    """
    Data augmentation pipeline
    
    Tuning augmentation for better generalization:
        - More augmentation = better generalization but slower training
        - Less augmentation = faster but may overfit
        
    Adjust probabilities (p=X):
        - Increase p if model overfits (memorizes training data)
        - Decrease p if training is too noisy/unstable
        - Current settings are moderate and work well
    
    Add more augmentations for robustness:
        - A.ElasticTransform() for sonar distortions
        - A.GridDistortion() for curved surfaces
        - A.CoarseDropout() for occlusions
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(p=0.2),  # Use default variance
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE Loss for segmentation
    
    Tuning the loss weights:
        - dice_weight controls overlap/shape matching (good for small objects)
        - bce_weight controls pixel-wise accuracy
        - Default (0.7, 0.3) works well for most cases
        - If nets are too fragmented: increase dice_weight to 0.8-0.9
        - If boundaries are imprecise: increase bce_weight to 0.4-0.5
        - For very small nets: try (0.9, 0.1) to prioritize shape
    """
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU metric"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def show_range_preview(data_dirs, max_range_display=10.0, original_range=20.0, image_size=256, debug_mode=False, mask_threshold=10.0):
    """Show preview of data at different ranges before training.
    
    Args:
        data_dirs: List of run directories
        max_range_display: Range to display (e.g., 10.0 for zoomed, 20.0 for full)
        original_range: Original collection range (e.g., 20.0)
        image_size: Output image size
        debug_mode: If True, also show all-white sonar for geometric debugging
        mask_threshold: Minimum sonar intensity for mask filtering
    
    Returns:
        True if user confirms, False to cancel
    """
    # Find first frame
    expanded_dirs = []
    for pattern in data_dirs:
        expanded_dirs.extend(glob.glob(pattern))
    
    if not expanded_dirs:
        print("❌ No directories found")
        return False
    
    frames_dir = Path(expanded_dirs[0]) / 'frames'
    frame_files = sorted(frames_dir.glob('frame_*.npz'))
    
    if not frame_files:
        print("❌ No frames found")
        return False
    
    # Load first frame
    data = np.load(frame_files[0])
    sonar_img = data['sonar_image']
    ground_truth = data['ground_truth']
    data.close()
    
    # Helper to convert polar to cartesian
    def polar_to_cart(polar, max_r, orig_r, is_mask=False):
        r_bins, n_beams = polar.shape
        fov_deg = 120.0
        half_width = max_r * np.sin(np.radians(fov_deg / 2))
        x = np.linspace(-half_width, half_width, image_size)
        y = np.linspace(0, max_r, image_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(X, Y)
        fov_rad = np.radians(fov_deg)
        beam_idx = (Theta + fov_rad / 2) / fov_rad * (n_beams - 1)
        range_idx = (R / orig_r) * (r_bins - 1)
        beam_idx = np.clip(beam_idx, 0, n_beams - 1).astype(np.float32)
        range_idx = np.clip(range_idx, 0, r_bins - 1).astype(np.float32)
        interp_method = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        cart = cv2.remap(polar.astype(np.float32), beam_idx, range_idx, 
                         interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Mask out regions outside cone
        cart[(R > max_r) | (Theta < -fov_rad/2) | (Theta > fov_rad/2)] = 0
        return cart
    
    # Convert at both ranges
    img_20m = polar_to_cart(sonar_img, 20.0, original_range, is_mask=False)
    img_zoomed = polar_to_cart(sonar_img, max_range_display, original_range, is_mask=False)
    
    # Convert masks using nearest-neighbor
    mask_polar = (ground_truth == 1).astype(np.float32)
    mask_20m = polar_to_cart(mask_polar, 20.0, original_range, is_mask=True)
    mask_zoomed = polar_to_cart(mask_polar, max_range_display, original_range, is_mask=True)
    
    # Apply signal filtering to masks
    if mask_threshold > 0:
        mask_20m_filtered = mask_20m * (img_20m > mask_threshold).astype(np.float32)
        mask_zoomed_filtered = mask_zoomed * (img_zoomed > mask_threshold).astype(np.float32)
    else:
        mask_20m_filtered = mask_20m
        mask_zoomed_filtered = mask_zoomed
    
    # Normalize for display
    img_20m_norm = np.clip(img_20m / (np.max(img_20m) + 1e-8), 0, 1)
    img_zoomed_norm = np.clip(img_zoomed / (np.max(img_zoomed) + 1e-8), 0, 1)
    
    # Create thresholded versions for debug mode (removed, using mask comparison instead)
    
    # Create visualization
    if debug_mode:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Row 1: Sonar images (actual)
    axes[0, 0].imshow(img_20m_norm, cmap='gray')
    axes[0, 0].set_title(f'Sonar @ 20m range (actual)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_zoomed_norm, cmap='gray')
    axes[0, 1].set_title(f'Sonar @ {max_range_display}m range (ZOOMED)', fontsize=11, fontweight='bold', color='blue')
    axes[0, 1].axis('off')
    
    if debug_mode:
        # Row 1 Col 3 & 4: Mask comparison (before and after filtering)
        axes[0, 2].imshow(mask_zoomed, cmap='gray')
        axes[0, 2].set_title(f'Mask BEFORE filtering (raw GT)', fontsize=11, fontweight='bold', color='orange')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(mask_zoomed_filtered, cmap='gray')
        axes[0, 3].set_title(f'Mask AFTER filtering (> {mask_threshold:.0f})', fontsize=11, fontweight='bold', color='green')
        axes[0, 3].axis('off')
    
    # Row 2: Masks (signal-filtered) with sonar overlay
    axes[1, 0].imshow(img_20m_norm, cmap='gray', alpha=0.6)
    axes[1, 0].imshow(mask_20m_filtered, cmap='hot', alpha=0.4)
    if mask_threshold > 0:
        axes[1, 0].set_title(f'Net mask @ 20m (filtered > {mask_threshold:.0f})', fontsize=11, fontweight='bold')
    else:
        axes[1, 0].set_title(f'Net mask @ 20m range', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_zoomed_norm, cmap='gray', alpha=0.6)
    axes[1, 1].imshow(mask_zoomed_filtered, cmap='hot', alpha=0.4)
    if mask_threshold > 0:
        axes[1, 1].set_title(f'Net mask @ {max_range_display}m (filtered > {mask_threshold:.0f}, ZOOMED)', fontsize=11, fontweight='bold', color='blue')
    else:
        axes[1, 1].set_title(f'Net mask @ {max_range_display}m range (ZOOMED)', fontsize=11, fontweight='bold', color='blue')
    axes[1, 1].axis('off')
    
    if debug_mode:
        # Row 2 Col 3 & 4: Overlays for comparison
        axes[1, 2].imshow(img_zoomed_norm, cmap='gray', alpha=0.6)
        axes[1, 2].imshow(mask_zoomed, cmap='hot', alpha=0.4)
        axes[1, 2].set_title(f'BEFORE: Raw GT on sonar', fontsize=11, fontweight='bold', color='orange')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(img_zoomed_norm, cmap='gray', alpha=0.6)
        axes[1, 3].imshow(mask_zoomed_filtered, cmap='hot', alpha=0.4)
        axes[1, 3].set_title(f'AFTER: Filtered GT on sonar', fontsize=11, fontweight='bold', color='green')
        axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Ask for confirmation
    if max_range_display < original_range:
        zoom_factor = original_range / max_range_display
        info_str = f"\n👀 Training at {max_range_display}m range ({zoom_factor:.1f}x zoom, {zoom_factor**2:.1f}x detail)\n"
    else:
        info_str = f"\n👀 Training at {max_range_display}m range (full view)\n"
    
    response = input(f"{info_str}Proceed with training? (y/n): ")
    return response.lower() == 'y'


def train_model(data_dirs=None, train_dirs=None, val_dirs=None, data_dir=None, epochs=50, batch_size=4, lr=1e-4, 
                sequence_length=5, stride=2, encoder='resnet34', hidden_dim=64,
                image_size=256, max_range=20.0, original_range=20.0, num_workers=2, use_amp=True, accumulation_steps=1,
                max_frames_per_run=None, max_sequences=None, resume_from=None, skip_preview=False, debug_preview=False, mask_threshold=10.0, preview_image_size=None, max_frames_list=None):
    """
    Train temporal segmentation model
    
    Key Tuning Tips:
        - If training is too slow: decrease image_size, sequence_length, or use max_frames_per_run
        - If OOM errors: decrease batch_size, image_size, or sequence_length
        - If quality is poor: increase image_size, sequence_length, epochs, or use better encoder
        - If overfitting: use more data augmentation, decrease model size, add dropout
        - If underfitting: increase model capacity (encoder, hidden_dim), train longer, check learning rate
    
    Performance vs Quality tradeoffs:
        Fast iteration:     image_size=256, sequence_length=3, batch_size=8, max_sequences=200
        Production training: image_size=256, sequence_length=3, batch_size=8, full dataset
        Best quality:       image_size=320, sequence_length=5, encoder='resnet50', epochs=100
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Mixed precision: {use_amp and device.type == 'cuda'}")
    print(f"Num workers: {num_workers}")
    if max_frames_per_run:
        print(f"Max frames per run: {max_frames_per_run}")
    if max_sequences:
        print(f"Max sequences: {max_sequences}")
    
    # Show preview if not skipped
    if data_dirs and not skip_preview:
        # Use preview_image_size if specified, otherwise use training image_size
        preview_size = preview_image_size if preview_image_size is not None else image_size
        
        print(f"\n" + "="*70)
        print(f"TRAINING PREVIEW")
        print(f"="*70)
        print(f"Display range: {max_range}m")
        print(f"Original range: {original_range}m")
        if max_range < original_range:
            print(f"Zoom factor: {original_range/max_range:.1f}x ({(original_range/max_range)**2:.1f}x detail)")
        if mask_threshold > 0:
            print(f"Mask filtering: Only pixels with signal > {mask_threshold} marked as net")
        else:
            print(f"Mask filtering: DISABLED (all ground truth pixels included)")
        print(f"Preview resolution: {preview_size}x{preview_size}" + (" (DEBUG MODE)" if preview_size != image_size else " (same as training)"))
        if preview_size != image_size:
            print(f"Training resolution: {image_size}x{image_size}")
        print(f"="*70)
        
        if not show_range_preview(data_dirs, max_range, original_range, preview_size, debug_preview, mask_threshold):
            print("\n❌ Training cancelled by user")
            return None
    
    print(f"\nStarting training...")
    print(f"="*70 + "\n")
    
    # Create datasets
    if train_dirs and val_dirs:
        # Explicit train/val split provided
        print("Loading NPZ sequences with explicit train/val split...")
        train_dirs_expanded = []
        val_dirs_expanded = []
        for pattern in train_dirs:
            train_dirs_expanded.extend(glob.glob(pattern))
        for pattern in val_dirs:
            val_dirs_expanded.extend(glob.glob(pattern))
        
        print(f"Train dirs: {[Path(d).name for d in train_dirs_expanded]}")
        print(f"Val dirs: {[Path(d).name for d in val_dirs_expanded]}")
        
        # Calculate validation split for max_sequences
        max_train_sequences = int(0.8 * max_sequences) if max_sequences else None
        max_val_sequences = max_sequences - max_train_sequences if max_sequences else None
        
        # Split max_frames_list if provided
        train_frames_list = None
        val_frames_list = None
        if max_frames_list:
            num_train = len(train_dirs_expanded)
            train_frames_list = max_frames_list[:num_train] if len(max_frames_list) >= num_train else max_frames_list
            val_frames_list = max_frames_list[num_train:] if len(max_frames_list) > num_train else None
        
        train_dataset = SonarSequenceDataset(train_dirs_expanded, sequence_length, stride, get_transforms(True), image_size=image_size, max_range=max_range, original_range_m=original_range, max_frames_per_run=max_frames_per_run, max_sequences=max_train_sequences, mask_threshold=mask_threshold, max_frames_list=train_frames_list)
        val_dataset = SonarSequenceDataset(val_dirs_expanded, sequence_length, stride, get_transforms(False), image_size=image_size, max_range=max_range, original_range_m=original_range, max_frames_per_run=max_frames_per_run, max_sequences=max_val_sequences, mask_threshold=mask_threshold, max_frames_list=val_frames_list)
    
    elif data_dirs:
        # Training on NPZ sequences from simulation (80/20 split from EACH directory)
        print("Loading NPZ sequences from simulation runs...")
        all_dirs = []
        for pattern in data_dirs:
            all_dirs.extend(glob.glob(pattern))
        
        # Load sequences from each directory and split 80/20 per directory
        all_train_sequences = []
        all_val_sequences = []
        
        for dir_idx, data_dir in enumerate(all_dirs):
            data_path = Path(data_dir)
            frames_dir = data_path / 'frames'
            
            if not frames_dir.exists():
                print(f"Warning: {frames_dir} does not exist, skipping")
                continue
            
            frame_files = sorted(frames_dir.glob('frame_*.npz'))
            
            # Determine frame limit for this directory
            if max_frames_list and dir_idx < len(max_frames_list):
                frame_limit = max_frames_list[dir_idx]
            else:
                frame_limit = max_frames_per_run
            
            # Limit frames per run
            if frame_limit:
                frame_files = frame_files[:frame_limit]
            
            # Create sequences with stride
            dir_sequences = []
            for i in range(0, len(frame_files) - sequence_length + 1, stride):
                seq = frame_files[i:i + sequence_length]
                if len(seq) == sequence_length:
                    dir_sequences.append(seq)
            
            # Split this directory's sequences 80/20
            num_train = int(0.8 * len(dir_sequences))
            train_seqs = dir_sequences[:num_train]
            val_seqs = dir_sequences[num_train:]
            
            all_train_sequences.extend(train_seqs)
            all_val_sequences.extend(val_seqs)
            
            print(f"  {data_path.name}: {len(frame_files)} frames → {len(dir_sequences)} sequences ({len(train_seqs)} train, {len(val_seqs)} val)")
        
        # Create parent dataset just for parameters (not used for iteration)
        parent_params = SonarSequenceDataset.__new__(SonarSequenceDataset)
        parent_params.sequence_length = sequence_length
        parent_params.stride = stride
        parent_params.max_range = max_range
        parent_params.original_range_m = original_range
        parent_params.fov = 120.0
        parent_params.image_size = image_size
        parent_params.mask_threshold = mask_threshold
        parent_params.transform = None
        parent_params.polar_to_cartesian = SonarSequenceDataset.polar_to_cartesian.__get__(parent_params, SonarSequenceDataset)
        
        # Create train and val datasets
        train_dataset = PreSplitSonarDataset(all_train_sequences, parent_params)
        train_dataset.parent.transform = get_transforms(True)
        
        val_dataset = PreSplitSonarDataset(all_val_sequences, parent_params)
        val_dataset.parent.transform = get_transforms(False)
        
        print(f"\nTotal: {len(all_train_sequences)} train sequences, {len(all_val_sequences)} val sequences")
    
    elif data_dir:
        # Training on augmented PNG data
        print("Loading augmented PNG sequences...")
        train_dataset = AugmentedSequenceDataset(data_dir, 'train', sequence_length, stride, get_transforms(True))
        val_dataset = AugmentedSequenceDataset(data_dir, 'val', sequence_length, stride, get_transforms(False))
    else:
        error_msg = "Must provide either:\n"
        error_msg += "  --data_dirs <patterns>  (auto 80/20 train/val split)\n"
        error_msg += "  --train_dirs <patterns> AND --val_dirs <patterns>  (explicit split)\n"
        error_msg += "  --data_dir <path>  (for pre-augmented PNG data)"
        raise ValueError(error_msg)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    
    print(f"Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")
    
    # Create model
    model = TemporalUNet(encoder_name=encoder, sequence_length=sequence_length, hidden_dim=hidden_dim).to(device)
    
    # Load checkpoint if resuming/fine-tuning
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        print(f"\n🔄 Loading checkpoint from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded model from epoch {checkpoint.get('epoch', 0)}")
        print(f"   Previous val IoU: {checkpoint.get('val_iou', 0):.4f}")
        print("   Starting fine-tuning...\n")
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_iou = 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            # Mixed precision training
            if scaler:
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            with torch.no_grad():
                train_iou += calculate_iou(outputs, masks)
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                if scaler:
                    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'sequence_length': sequence_length,
                'hidden_dim': hidden_dim,
                'encoder': encoder
            }, 'best_temporal_segmentation.pth')
            print(f"  ✅ Saved best model (IoU={val_iou:.4f})")
    
    print(f"\n✅ Training complete! Best Val IoU: {best_val_iou:.4f}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train temporal segmentation model')
    
    # Data
    parser.add_argument('--data_dirs', type=str, nargs='+',
                       help='Paths to simulation run directories (supports wildcards). Will be split 80/20 train/val')
    parser.add_argument('--train_dirs', type=str, nargs='+',
                       help='Explicit training directories (use with --val_dirs for custom splits)')
    parser.add_argument('--val_dirs', type=str, nargs='+',
                       help='Explicit validation directories (use with --train_dirs)')
    parser.add_argument('--data_dir', type=str,
                       help='Path to augmented training data directory (alternative to data_dirs)')
    
    # Model Architecture
    parser.add_argument('--encoder', type=str, default='resnet34',
                       choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet-b0', 'efficientnet-b4'],
                       help='Encoder backbone. Speed: resnet18 (fast) > resnet34 (balanced) > resnet50 (slow, best quality)')
    parser.add_argument('--sequence_length', type=int, default=5,
                       help='Frames per sequence. More = better temporal context but slower. Try: 3 (fast), 5 (balanced), 7 (slow)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='ConvLSTM hidden dimension. Higher = more capacity but slower. Try: 32, 64 (default), 128')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs. More = better convergence. Try: 10 (test), 30 (quick), 50 (good), 100 (best)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size per GPU. Larger = faster + smoother gradients. Try: 2 (OOM fallback), 4-8 (balanced), 16+ (if memory allows)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate. Default 1e-4 works well. Increase to 3e-4 for faster convergence or decrease to 5e-5 for fine-tuning')
    parser.add_argument('--stride', type=int, default=2,
                       help='Stride between sequences. Lower = more overlapping data. Try: 1 (max data), 2 (balanced), 5 (less data, faster)')
    
    # Performance Optimization
    parser.add_argument('--image_size', type=int, default=256,
                       help='Output image resolution. Larger = better detail but much slower. Try: 224 (fastest), 256 (fast), 320 (balanced), 400 (slow)')
    parser.add_argument('--max_range', type=float, default=20.0,
                       help='Display range in meters. Use < 20.0 to zoom in (e.g., 10.0 for 2x zoom). Default: 20.0')
    parser.add_argument('--original_range', type=float, default=20.0,
                       help='Original range the data was collected at in meters. Default: 20.0')
    parser.add_argument('--mask_threshold', type=float, default=10.0,
                       help='Minimum sonar intensity for pixel to be marked as net. Use 0 to disable filtering. Default: 10.0')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Parallel data loading workers. More = faster loading. Try: 0 (debug), 2-4 (normal), 8+ (many cores). Set to 0 if errors occur')
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable mixed precision (FP16) training. AMP gives ~2x speedup on modern GPUs with minimal quality loss. Disable only for debugging')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps. Simulates larger batch size without more memory. Try: 2-4 if OOM errors. Effective batch = batch_size * accumulation_steps')
    
    # Data Subsampling (for faster iteration during development)
    parser.add_argument('--max_frames_per_run', type=int, default=None,
                       help='Limit frames per run directory. Use for quick iterations. Try: 30 (very fast test), 50-100 (development), None (full dataset)')
    parser.add_argument('--max_frames_list', type=int, nargs='+', default=None,
                       help='Per-directory frame limits (space-separated integers, one per data source). Example: --max_frames_list 500 2000 1000. Overrides --max_frames_per_run for each specified directory.')
    parser.add_argument('--max_sequences', type=int, default=None,
                       help='Cap total training sequences. Useful for quick experiments. Try: 100 (fast test), 500 (development), None (all data)')
    
    # Fine-tuning
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume/fine-tune from (e.g., best_temporal_segmentation.pth)')
    
    # Preview
    parser.add_argument('--skip_preview', action='store_true',
                       help='Skip the data range preview before training')
    parser.add_argument('--debug_preview', action='store_true',
                       help='Show all-white sonar images in preview for geometric debugging')
    parser.add_argument('--preview_image_size', type=int, default=None,
                       help='Image resolution for preview only (e.g., 100 for debugging). Default: same as --image_size')
    
    args = parser.parse_args()
    
    # Pass debug_preview flag to train_model
    train_model(
        data_dirs=args.data_dirs,
        train_dirs=args.train_dirs,
        val_dirs=args.val_dirs,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sequence_length=args.sequence_length,
        stride=args.stride,
        debug_preview=args.debug_preview,
        mask_threshold=args.mask_threshold,
        preview_image_size=args.preview_image_size,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        image_size=args.image_size,
        max_range=args.max_range,
        original_range=args.original_range,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        accumulation_steps=args.accumulation_steps,
        max_frames_per_run=args.max_frames_per_run,
        max_sequences=args.max_sequences,
        resume_from=args.resume_from,
        skip_preview=args.skip_preview,
        max_frames_list=args.max_frames_list
    )
