"""
Training script for semantic segmentation of nets in sonar imagery.

Requirements:
    pip install torch torchvision segmentation-models-pytorch albumentations

Usage:
    python train_segmentation.py --data_dir ../training_data --epochs 50
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


class SonarDataset(Dataset):
    """Load sonar images and masks from PNG files"""
    def __init__(self, data_dir, split='train', transform=None):
        self.img_dir = Path(data_dir) / split / 'images'
        self.mask_dir = Path(data_dir) / split / 'masks'
        self.files = sorted(self.img_dir.glob('*.png'))
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = self.mask_dir / img_path.name
        
        # Load as grayscale
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        # Convert to 3-channel (required by pretrained models)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.unsqueeze(0)  # Add channel dim


def get_transforms(train=True):
    """Data augmentation for sonar images"""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class DiceLoss(nn.Module):
    """Dice Loss for handling class imbalance"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def train_model(data_dir, epochs=50, batch_size=8, lr=1e-4):
    """Train U-Net segmentation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please run export_training_data() in the notebook first to generate training data."
        )
    
    train_img_dir = data_path / 'train' / 'images'
    if not train_img_dir.exists() or len(list(train_img_dir.glob('*.png'))) == 0:
        raise FileNotFoundError(
            f"No training images found in {train_img_dir}\n"
            f"Please run export_training_data() in the notebook first.\n"
            f"Example: export_training_data(num_frames=1000, train_split=0.8)"
        )
    
    # Create datasets
    train_dataset = SonarDataset(data_dir, 'train', get_transforms(train=True))
    val_dataset = SonarDataset(data_dir, 'val', get_transforms(train=False))
    
    # Use num_workers=0 on macOS to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Create model (U-Net with ResNet34 encoder)
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None  # We'll apply sigmoid in loss
    ).to(device)
    
    # Loss and optimizer
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_net_segmentation.pth')
            print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")
    
    print("\n✅ Training complete!")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train net segmentation model')
    parser.add_argument('--data_dir', type=str, default='training_data',
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    args = parser.parse_args()
    
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr)
