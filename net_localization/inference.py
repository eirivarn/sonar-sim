"""
Inference script for trained segmentation model.

Usage:
    python inference.py --model best_net_segmentation.pth --image_dir test_images/
    
    # With temporal smoothing (recommended for video sequences)
    python inference.py --model best_net_segmentation.pth --image_dir test_images/ \
        --temporal_smoothing --window_size 5 --smooth_method median
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
from temporal_smoothing import TemporalSmoother, ProbabilityTemporalSmoother


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


def predict_mask(model, image, device='cpu', smoother=None):
    """
    Predict segmentation mask for a single image.
    
    Args:
        model: Trained model
        image: Grayscale image (H, W) or RGB image (H, W, 3)
        device: torch device
        smoother: Optional TemporalSmoother for temporal filtering
    
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
        prob_map = output[0, 0].cpu().numpy()  # Probability map (0-1)
    
    # Apply temporal smoothing if available
    if smoother is not None:
        if isinstance(smoother, ProbabilityTemporalSmoother):
            # Smooth probabilities, then threshold
            mask = smoother.update_and_threshold(prob_map)
        else:
            # Threshold first, then smooth
            binary = (prob_map > 0.5).astype(np.uint8) * 255
            mask = smoother.update(binary)
    else:
        # No smoothing
        mask = (prob_map > 0.5).astype(np.uint8) * 255
    
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


def process_directory(model, image_dir, output_dir=None, device='cpu', 
                     temporal_smoothing=False, smooth_method='median', 
                     window_size=5, alpha=0.3):
    """Process all images in a directory"""
    image_dir = Path(image_dir)
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize temporal smoother if requested
    smoother = None
    if temporal_smoothing:
        smoother = ProbabilityTemporalSmoother(
            window_size=window_size,
            method=smooth_method,
            alpha=alpha,
            threshold=0.5
        )
        print(f"🔄 Temporal smoothing enabled: {smooth_method}, window={window_size}")
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        # Load image
        image = np.array(Image.open(img_path).convert('L'))
        
        # Predict with optional temporal smoothing
        mask = predict_mask(model, image, device, smoother)
        
        # Extract lines
        lines = extract_lines_from_mask(mask)
        
        print(f"  {img_path.name}: {len(lines)} lines extracted")
        
        # Save visualization
        if output_dir:
            save_path = output_dir / f"{img_path.stem}_result.png"
            visualize_prediction(image, mask, lines, save_path)
    
    if temporal_smoothing and smoother:
        print(f"  ℹ️  Smoother warmed up: {smoother.is_warmed_up()}")
    
    print(f"\n✅ Processed {len(image_files)} images")


def process_video_frames(model, frame_dir, output_video=None, device='cpu',
                        temporal_smoothing=True, smooth_method='median',
                        window_size=5, alpha=0.3, fps=30):
    """
    Process sequential video frames with temporal smoothing.
    
    Args:
        model: Trained model
        frame_dir: Directory with frames (e.g., frame_000000.npz)
        output_video: Path to save output video (optional)
        device: torch device
        temporal_smoothing: Enable temporal smoothing
        smooth_method: 'median', 'mean', or 'exponential'
        window_size: Temporal window size
        alpha: For exponential smoothing
        fps: Output video frame rate
    """
    from pathlib import Path
    
    frame_dir = Path(frame_dir)
    frame_files = sorted(frame_dir.glob('frame_*.npz'))
    
    if not frame_files:
        print(f"❌ No frame files found in {frame_dir}")
        return
    
    # Initialize smoother
    smoother = None
    if temporal_smoothing:
        smoother = ProbabilityTemporalSmoother(
            window_size=window_size,
            method=smooth_method,
            alpha=alpha,
            threshold=0.5
        )
        print(f"🔄 Temporal smoothing: {smooth_method}, window={window_size}")
    
    # Setup video writer if requested
    video_writer = None
    if output_video:
        # Load first frame to get dimensions
        data = np.load(frame_files[0])
        sonar = data['sonar_image']
        H, W = sonar.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (W * 2, H))
        print(f"📹 Writing video to {output_video}")
    
    print(f"Processing {len(frame_files)} frames...")
    
    for frame_path in frame_files:
        # Load NPZ frame
        data = np.load(frame_path)
        sonar_polar = data['sonar_image']
        
        # Convert to Cartesian for display (simplified)
        # You may want to use your actual polar_to_cartesian function
        sonar_display = cv2.resize(sonar_polar, (400, 400))
        
        # Predict with temporal smoothing
        mask = predict_mask(model, sonar_display, device, smoother)
        
        # Create visualization frame
        if video_writer:
            # Side-by-side: original + mask overlay
            vis_img = cv2.cvtColor(sonar_display, cv2.COLOR_GRAY2BGR)
            vis_mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            combined = np.hstack([vis_img, vis_mask])
            video_writer.write(combined)
    
    if video_writer:
        video_writer.release()
        print(f"✅ Video saved to {output_video}")
    
    print(f"✅ Processed {len(frame_files)} frames")


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
    
    # Temporal smoothing options
    parser.add_argument('--temporal_smoothing', action='store_true',
                        help='Enable temporal smoothing (for video sequences)')
    parser.add_argument('--smooth_method', type=str, default='median',
                        choices=['median', 'mean', 'exponential'],
                        help='Smoothing method (median recommended for sonar)')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Temporal window size (3-7 recommended)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Alpha for exponential smoothing (0.1-0.5)')
    
    # Video processing mode
    parser.add_argument('--video_mode', action='store_true',
                        help='Process as video frames (expects frame_*.npz files)')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Output video path (for video mode)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video frame rate')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    
    # Process images
    if args.video_mode:
        process_video_frames(
            model, args.image_dir, args.output_video, device,
            temporal_smoothing=args.temporal_smoothing,
            smooth_method=args.smooth_method,
            window_size=args.window_size,
            alpha=args.alpha,
            fps=args.fps
        )
    else:
        process_directory(
            model, args.image_dir, args.output_dir, device,
            temporal_smoothing=args.temporal_smoothing,
            smooth_method=args.smooth_method,
            window_size=args.window_size,
            alpha=args.alpha
        )
