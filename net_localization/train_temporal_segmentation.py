"""
Temporal segmentation training for sonar net detection.

Uses ConvLSTM + U-Net architecture to leverage temporal information from video sequences.

Requirements:
    pip install torch torchvision segmentation-models-pytorch albumentations

Usage:
    # Train on NPZ sequences from simulation
    python train_temporal_segmentation.py --data_dirs ../simulation/data/runs/temp_augmentation_* --epochs 50 --sequence_length 5
    
    # Train on augmented PNG data
    python train_temporal_segmentation.py --data_dir training_data_augmented --epochs 50 --sequence_length 3
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
    def __init__(self, data_dirs, sequence_length=5, stride=2, transform=None, max_range=20.0, fov=120.0):
        """
        Args:
            data_dirs: List of paths to simulation run directories
            sequence_length: Number of frames per sequence
            stride: Stride between sequences (stride=1 means overlapping sequences)
            transform: Albumentations transforms
            max_range: Sonar max range in meters
            fov: Sonar field of view in degrees
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.max_range = max_range
        self.fov = fov
        
        # Collect all frame files from all directories
        self.sequences = []
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            frames_dir = data_path / 'frames'
            
            if not frames_dir.exists():
                print(f"Warning: {frames_dir} does not exist, skipping")
                continue
            
            frame_files = sorted(frames_dir.glob('frame_*.npz'))
            
            # Create sequences with stride
            for i in range(0, len(frame_files) - sequence_length + 1, stride):
                seq = frame_files[i:i + sequence_length]
                if len(seq) == sequence_length:
                    self.sequences.append(seq)
        
        print(f"Created {len(self.sequences)} sequences from {len(data_dirs)} runs")
    
    def polar_to_cartesian(self, polar_image, output_size=400):
        """Convert polar sonar image to Cartesian coordinates"""
        r_bins, n_beams = polar_image.shape
        half_width = self.max_range * np.sin(np.radians(self.fov / 2))
        x = np.linspace(-half_width, half_width, output_size)
        y = np.linspace(0, self.max_range, output_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(X, Y)
        fov_rad = np.radians(self.fov)
        beam_idx = (Theta + fov_rad / 2) / fov_rad * (n_beams - 1)
        range_idx = (R / self.max_range) * (r_bins - 1)
        beam_idx = np.clip(beam_idx, 0, n_beams - 1).astype(np.float32)
        range_idx = np.clip(range_idx, 0, r_bins - 1).astype(np.float32)
        cart = cv2.remap(polar_image.astype(np.float32), beam_idx, range_idx, 
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
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
            img_cart = self.polar_to_cartesian(sonar_img)
            mask_polar = (ground_truth == 1).astype(np.float32)  # Net material ID = 1
            mask_cart = self.polar_to_cartesian(mask_polar)
            mask_cart = (mask_cart > 0.5).astype(np.float32)
            
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
    """Data augmentation"""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class CombinedLoss(nn.Module):
    """Combined Dice + BCE Loss"""
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


def train_model(data_dirs=None, data_dir=None, epochs=50, batch_size=4, lr=1e-4, 
                sequence_length=5, stride=2, encoder='resnet34', hidden_dim=64):
    """Train temporal segmentation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    if data_dirs:
        # Training on NPZ sequences from simulation
        print("Loading NPZ sequences from simulation runs...")
        all_dirs = []
        for pattern in data_dirs:
            all_dirs.extend(glob.glob(pattern))
        
        num_train = int(0.8 * len(all_dirs))
        train_dirs = all_dirs[:num_train]
        val_dirs = all_dirs[num_train:]
        
        train_dataset = SonarSequenceDataset(train_dirs, sequence_length, stride, get_transforms(True))
        val_dataset = SonarSequenceDataset(val_dirs, sequence_length, stride, get_transforms(False))
    
    elif data_dir:
        # Training on augmented PNG data
        print("Loading augmented PNG sequences...")
        train_dataset = AugmentedSequenceDataset(data_dir, 'train', sequence_length, stride, get_transforms(True))
        val_dataset = AugmentedSequenceDataset(data_dir, 'val', sequence_length, stride, get_transforms(False))
    else:
        raise ValueError("Must provide either --data_dirs or --data_dir")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")
    
    # Create model
    model = TemporalUNet(encoder_name=encoder, sequence_length=sequence_length, hidden_dim=hidden_dim).to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_iou = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, masks = images.to(device), masks.to(device)
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
                       help='Paths to simulation run directories (supports wildcards)')
    parser.add_argument('--data_dir', type=str,
                       help='Path to augmented training data directory (alternative to data_dirs)')
    
    # Model
    parser.add_argument('--encoder', type=str, default='resnet34',
                       choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet-b0', 'efficientnet-b4'],
                       help='Encoder backbone')
    parser.add_argument('--sequence_length', type=int, default=5,
                       help='Number of frames per sequence')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for ConvLSTM')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (sequences)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--stride', type=int, default=2,
                       help='Stride between sequences')
    
    args = parser.parse_args()
    
    train_model(
        data_dirs=args.data_dirs,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sequence_length=args.sequence_length,
        stride=args.stride,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim
    )
