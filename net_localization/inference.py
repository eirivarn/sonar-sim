"""
Inference script for trained segmentation model.

Usage:
    python inference.py --model best_net_segmentation.pth --image_dir test_images/
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import argparse
import matplotlib.pyplot as plt


def load_model(model_path, device='cpu'):
    """Load trained segmentation model"""
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model


def get_inference_transform():
    """Transform for inference"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def predict_mask(model, image, device='cpu'):
    """
    Predict segmentation mask for a single image.
    
    Args:
        model: Trained model
        image: Grayscale image (H, W) or RGB image (H, W, 3)
        device: torch device
    
    Returns:
        Binary mask (H, W) with values 0-255
    """
    # Ensure 3-channel
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    
    # Transform
    transform = get_inference_transform()
    img_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        mask = (output[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    
    return mask


def extract_lines_from_mask(mask, min_length=20):
    """
    Convert binary mask to line segments using skeleton + contour fitting.
    
    Args:
        mask: Binary mask (255 = net, 0 = background)
        min_length: Minimum line length in pixels
    
    Returns:
        lines: Array of line segments [(x1, y1, x2, y2), ...]
    """
    from scipy import ndimage
    
    # Morphological skeleton
    def morphological_skeleton(binary):
        skeleton = np.zeros_like(binary, dtype=bool)
        element = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)
        
        eroded = binary.astype(bool)
        while np.any(eroded):
            opened = ndimage.binary_opening(eroded, structure=element)
            skeleton |= eroded & ~opened
            eroded = ndimage.binary_erosion(eroded, structure=element)
        
        return skeleton
    
    # Thin to skeleton
    mask_binary = (mask > 127).astype(np.uint8)
    skeleton = morphological_skeleton(mask_binary)
    skeleton_u8 = (skeleton * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(skeleton_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fit lines to contours
    lines = []
    for contour in contours:
        if len(contour) < 5:
            continue
        
        # Fit line using least squares
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Get endpoints from contour extent
        pts = contour.reshape(-1, 2)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        
        if length < min_length:
            continue
        
        # Compute line endpoints
        t1 = -length / 2
        t2 = length / 2
        x1, y1 = x + vx * t1, y + vy * t1
        x2, y2 = x + vx * t2, y + vy * t2
        
        lines.append([x1[0], y1[0], x2[0], y2[0]])
    
    return np.array(lines) if lines else np.array([]).reshape(0, 4)


def visualize_prediction(image, mask, lines=None, save_path=None):
    """Visualize prediction results"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    # Predicted mask
    ax[1].imshow(image, cmap='gray', alpha=0.5)
    ax[1].imshow(mask, cmap='hot', alpha=0.5)
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')
    
    # Extracted lines
    ax[2].imshow(image, cmap='gray', alpha=0.5)
    if lines is not None and len(lines) > 0:
        for line in lines:
            ax[2].plot([line[0], line[2]], [line[1], line[3]], 
                      'lime', linewidth=2, alpha=0.8)
    ax[2].set_title(f'Extracted Lines ({0 if lines is None else len(lines)})')
    ax[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def process_directory(model, image_dir, output_dir=None, device='cpu'):
    """Process all images in a directory"""
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        # Load image
        image = np.array(Image.open(img_path).convert('L'))
        
        # Predict
        mask = predict_mask(model, image, device)
        
        # Extract lines
        lines = extract_lines_from_mask(mask)
        
        print(f"  {img_path.name}: {len(lines)} lines extracted")
        
        # Save visualization
        if output_dir:
            save_path = output_dir / f"{img_path.stem}_result.png"
            visualize_prediction(image, mask, lines, save_path)
    
    print(f"\n✅ Processed {len(image_files)} images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cpu/cuda)')
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    
    # Process images
    process_directory(model, args.image_dir, args.output_dir, device)
