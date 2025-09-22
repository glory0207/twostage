#!/usr/bin/env python3
"""
配准结果差异可视化工具
用于比较stage1和stage2配准结果的差异
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

def load_image_pair(stage1_path, stage2_path):
    """加载一对配准图像"""
    stage1_img = cv2.imread(str(stage1_path))
    stage2_img = cv2.imread(str(stage2_path))
    
    if stage1_img is None or stage2_img is None:
        return None, None
    
    # BGR to RGB
    stage1_img = cv2.cvtColor(stage1_img, cv2.COLOR_BGR2RGB)
    stage2_img = cv2.cvtColor(stage2_img, cv2.COLOR_BGR2RGB)
    
    return stage1_img, stage2_img

def compute_difference_map(img1, img2):
    """计算两张图像的差异图"""
    # 转换为float进行计算
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    
    # 计算绝对差异
    diff = np.abs(img1_f - img2_f)
    
    # 归一化到0-255
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    return diff

def compute_intensity_difference(img1, img2):
    """计算强度差异（灰度）"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
    return diff

def create_overlay_image(img1, img2, alpha=0.5):
    """Create overlay image with different blend modes"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Method 1: Color-coded overlay (Red: Stage1, Green: Stage2)
    overlay = np.zeros_like(img1)
    overlay[:, :, 0] = gray1  # Red channel for stage1
    overlay[:, :, 1] = gray2  # Green channel for stage2
    overlay[:, :, 2] = np.minimum(gray1, gray2) * 0.5  # Blue for common areas
    
    return overlay

def create_advanced_overlay_visualizations(img1, img2):
    """Create multiple overlay visualization methods"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # 1. Color-coded overlay
    color_overlay = np.zeros_like(img1)
    color_overlay[:, :, 0] = gray1  # Red: Stage1
    color_overlay[:, :, 1] = gray2  # Green: Stage2
    color_overlay[:, :, 2] = np.minimum(gray1, gray2) * 0.5
    
    # 2. Difference overlay with heatmap
    diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
    
    # 3. Alpha blending
    alpha = 0.5
    blended = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
    
    # 4. Side-by-side with difference
    h, w = img1.shape[:2]
    side_by_side = np.zeros((h, w*2, 3), dtype=np.uint8)
    side_by_side[:, :w] = img1
    side_by_side[:, w:] = img2
    
    # Add vertical line separator
    side_by_side[:, w-2:w+2] = [255, 255, 0]  # Yellow line
    
    return color_overlay, diff_colored, blended, side_by_side

def create_checkerboard_comparison(img1, img2, square_size=50):
    """创建棋盘格对比图"""
    h, w = img1.shape[:2]
    result = img1.copy()
    
    # 创建棋盘格掩码
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            # 棋盘格模式
            if ((i // square_size) + (j // square_size)) % 2 == 1:
                end_i = min(i + square_size, h)
                end_j = min(j + square_size, w)
                result[i:end_i, j:end_j] = img2[i:end_i, j:end_j]
    
    return result

def create_edge_comparison(img1, img2):
    """创建边缘对比图"""
    # 计算边缘
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)
    
    # 创建彩色边缘对比
    edge_comparison = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    edge_comparison[:, :, 0] = edges1  # 红色：stage1边缘
    edge_comparison[:, :, 1] = edges2  # 绿色：stage2边缘
    edge_comparison[:, :, 2] = np.bitwise_and(edges1, edges2)  # 蓝色：共同边缘
    
    return edge_comparison

def create_comprehensive_comparison(stage1_img, stage2_img, pair_id, save_path):
    """Create comprehensive comparison visualization"""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # 1. Original image comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(stage1_img)
    ax1.set_title('Stage1 Registration Result', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(stage2_img)
    ax2.set_title('Stage2 Registration Result', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 2. Difference map
    diff_map = compute_difference_map(stage1_img, stage2_img)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(diff_map)
    ax3.set_title('Absolute Difference Map', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 3. Intensity difference heatmap
    intensity_diff = compute_intensity_difference(stage1_img, stage2_img)
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(intensity_diff, cmap='hot')
    ax4.set_title('Intensity Difference Heatmap', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # 4. Overlay image
    overlay = create_overlay_image(stage1_img, stage2_img)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(overlay)
    ax5.set_title('Color Overlay\n(Red:Stage1, Green:Stage2)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 5. Checkerboard comparison
    checkerboard = create_checkerboard_comparison(stage1_img, stage2_img)
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(checkerboard)
    ax6.set_title('Checkerboard Comparison', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 6. Edge comparison
    edge_comp = create_edge_comparison(stage1_img, stage2_img)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(edge_comp)
    ax7.set_title('Edge Comparison\n(Red:Stage1, Green:Stage2)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # 7. Local zoom comparison
    h, w = stage1_img.shape[:2]
    crop_size = min(h, w) // 3
    start_h, start_w = h//2 - crop_size//2, w//2 - crop_size//2
    end_h, end_w = start_h + crop_size, start_w + crop_size
    
    crop1 = stage1_img[start_h:end_h, start_w:end_w]
    crop2 = stage2_img[start_h:end_h, start_w:end_w]
    
    # Side-by-side local zoom
    combined_crop = np.concatenate([crop1, crop2], axis=1)
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(combined_crop)
    ax8.set_title('Local Zoom Comparison\n(Left:Stage1, Right:Stage2)', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    # Mark zoom regions on original images
    ax1.add_patch(Rectangle((start_w, start_h), crop_size, crop_size, 
                           linewidth=2, edgecolor='yellow', facecolor='none'))
    ax2.add_patch(Rectangle((start_w, start_h), crop_size, crop_size, 
                           linewidth=2, edgecolor='yellow', facecolor='none'))
    
    # 8. Statistics
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    # Calculate statistical metrics
    mse = np.mean((stage1_img.astype(np.float32) - stage2_img.astype(np.float32)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    mean_diff = np.mean(intensity_diff)
    max_diff = np.max(intensity_diff)
    
    # Calculate structural similarity (simplified)
    gray1 = cv2.cvtColor(stage1_img, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(stage2_img, cv2.COLOR_RGB2GRAY)
    correlation = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
    
    stats_text = f"""
    Image Pair: {pair_id}
    
    Difference Statistics:
    • Mean Squared Error (MSE): {mse:.2f}
    • Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB
    • Mean Intensity Difference: {mean_diff:.2f}
    • Max Intensity Difference: {max_diff:.2f}
    • Correlation Coefficient: {correlation:.4f}
    
    Color Legend:
    • Red regions: Stage1 unique features
    • Green regions: Stage2 unique features  
    • Yellow regions: Common features
    • Bright areas in heatmap: High difference regions
    """
    
    ax9.text(0.02, 0.98, stats_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Registration Results Comparison Analysis - {pair_id}', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'correlation': correlation
    }

def create_overlay_difference_visualization(stage1_img, stage2_img, pair_id, save_path):
    """Create specialized overlay difference visualization"""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Get advanced overlay visualizations
    color_overlay, diff_colored, blended, side_by_side = create_advanced_overlay_visualizations(stage1_img, stage2_img)
    
    # 1. Color-coded overlay (Red: Stage1, Green: Stage2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(color_overlay)
    ax1.set_title('Color-Coded Overlay\n(Red:Stage1, Green:Stage2, Blue:Common)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Difference heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(diff_colored)
    ax2.set_title('Difference Heatmap\n(Blue:Low, Red:High Difference)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Alpha blended
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(blended)
    ax3.set_title('Alpha Blended\n(50% Stage1 + 50% Stage2)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Side-by-side with separator
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(side_by_side)
    ax4.set_title('Side-by-Side Comparison\n(Left:Stage1, Right:Stage2)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Zoomed color overlay (center region)
    h, w = stage1_img.shape[:2]
    crop_size = min(h, w) // 2
    start_h, start_w = h//2 - crop_size//2, w//2 - crop_size//2
    end_h, end_w = start_h + crop_size, start_w + crop_size
    
    zoomed_overlay = color_overlay[start_h:end_h, start_w:end_w]
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(zoomed_overlay)
    ax5.set_title('Zoomed Color Overlay\n(Center Region)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. Zoomed difference heatmap
    zoomed_diff = diff_colored[start_h:end_h, start_w:end_w]
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(zoomed_diff)
    ax6.set_title('Zoomed Difference Heatmap\n(Center Region)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 7. Edge overlay comparison
    gray1 = cv2.cvtColor(stage1_img, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(stage2_img, cv2.COLOR_RGB2GRAY)
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)
    
    edge_overlay = np.zeros_like(stage1_img)
    edge_overlay[:, :, 0] = edges1  # Red: Stage1 edges
    edge_overlay[:, :, 1] = edges2  # Green: Stage2 edges
    edge_overlay[:, :, 2] = np.bitwise_and(edges1, edges2)  # Blue: Common edges
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(edge_overlay)
    ax7.set_title('Edge Overlay\n(Red:Stage1, Green:Stage2, Blue:Common)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # 8. Difference statistics visualization
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    # Calculate metrics
    intensity_diff = compute_intensity_difference(stage1_img, stage2_img)
    mse = np.mean((stage1_img.astype(np.float32) - stage2_img.astype(np.float32)) ** 2)
    mean_diff = np.mean(intensity_diff)
    max_diff = np.max(intensity_diff)
    std_diff = np.std(intensity_diff)
    
    # Create difference histogram
    diff_hist, bins = np.histogram(intensity_diff.flatten(), bins=50, range=(0, 255))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot histogram
    ax8_hist = fig.add_axes([0.77, 0.15, 0.2, 0.25])  # [left, bottom, width, height]
    ax8_hist.bar(bin_centers, diff_hist, width=bins[1]-bins[0], alpha=0.7, color='blue')
    ax8_hist.set_xlabel('Intensity Difference')
    ax8_hist.set_ylabel('Frequency')
    ax8_hist.set_title('Difference Distribution')
    
    # Add statistics text
    stats_text = f"""Overlay Analysis Statistics:

MSE: {mse:.2f}
Mean Diff: {mean_diff:.2f}
Max Diff: {max_diff:.2f}
Std Diff: {std_diff:.2f}

Interpretation:
• Red areas: Stage1 dominant
• Green areas: Stage2 dominant
• Yellow areas: Similar intensity
• Dark areas: Low intensity both stages
• Bright heatmap: High difference"""
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Overlay Difference Analysis - {pair_id}', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def analyze_all_pairs(input_dir, output_dir):
    """Analyze all registration pairs"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all stage1 images
    stage1_files = list(input_path.glob("*_stage1_registered.jpg"))
    
    if not stage1_files:
        print(f"No stage1 registration images found in {input_dir}")
        return
    
    print(f"Found {len(stage1_files)} registration pairs")
    
    all_stats = []
    
    for stage1_file in stage1_files:
        # Construct corresponding stage2 filename
        pair_id = stage1_file.stem.replace("_stage1_registered", "")
        stage2_file = input_path / f"{pair_id}_stage2_final.jpg"
        
        if not stage2_file.exists():
            print(f"Warning: Cannot find corresponding stage2 file: {stage2_file}")
            continue
        
        print(f"Processing registration pair: {pair_id}")
        
        # Load images
        stage1_img, stage2_img = load_image_pair(stage1_file, stage2_file)
        if stage1_img is None or stage2_img is None:
            print(f"Cannot load image pair: {pair_id}")
            continue
        
        # Create comprehensive comparison analysis
        output_file = output_path / f"{pair_id}_comparison.png"
        stats = create_comprehensive_comparison(stage1_img, stage2_img, pair_id, output_file)
        stats['pair_id'] = pair_id
        all_stats.append(stats)
        
        # Create specialized overlay difference visualization
        overlay_output_file = output_path / f"{pair_id}_overlay_analysis.png"
        create_overlay_difference_visualization(stage1_img, stage2_img, pair_id, overlay_output_file)
        
        print(f"Saved comparison images: {output_file.name} and {overlay_output_file.name}")
    
    # Generate overall statistical report
    if all_stats:
        create_summary_report(all_stats, output_path)
    
    print(f"\nAnalysis completed! Results saved in: {output_path}")

def create_summary_report(all_stats, output_path):
    """Create overall statistical report"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Registration Results Difference Statistical Report', fontsize=16, fontweight='bold')
    
    # Extract data
    pair_ids = [s['pair_id'] for s in all_stats]
    mse_values = [s['mse'] for s in all_stats]
    psnr_values = [s['psnr'] for s in all_stats if s['psnr'] != float('inf')]
    mean_diff_values = [s['mean_diff'] for s in all_stats]
    max_diff_values = [s['max_diff'] for s in all_stats]
    correlation_values = [s['correlation'] for s in all_stats]
    
    # 1. MSE distribution
    axes[0, 0].bar(range(len(pair_ids)), mse_values)
    axes[0, 0].set_title('Mean Squared Error (MSE)')
    axes[0, 0].set_xlabel('Image Pairs')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_xticks(range(len(pair_ids)))
    axes[0, 0].set_xticklabels(pair_ids, rotation=45)
    
    # 2. PSNR distribution
    if psnr_values:
        axes[0, 1].bar(range(len(psnr_values)), psnr_values)
        axes[0, 1].set_title('Peak Signal-to-Noise Ratio (PSNR)')
        axes[0, 1].set_xlabel('Image Pairs')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_xticks(range(len(psnr_values)))
        axes[0, 1].set_xticklabels([pair_ids[i] for i in range(len(psnr_values))], rotation=45)
    
    # 3. Mean difference
    axes[0, 2].bar(range(len(pair_ids)), mean_diff_values)
    axes[0, 2].set_title('Mean Intensity Difference')
    axes[0, 2].set_xlabel('Image Pairs')
    axes[0, 2].set_ylabel('Mean Difference')
    axes[0, 2].set_xticks(range(len(pair_ids)))
    axes[0, 2].set_xticklabels(pair_ids, rotation=45)
    
    # 4. Max difference
    axes[1, 0].bar(range(len(pair_ids)), max_diff_values)
    axes[1, 0].set_title('Maximum Intensity Difference')
    axes[1, 0].set_xlabel('Image Pairs')
    axes[1, 0].set_ylabel('Max Difference')
    axes[1, 0].set_xticks(range(len(pair_ids)))
    axes[1, 0].set_xticklabels(pair_ids, rotation=45)
    
    # 5. Correlation coefficient
    axes[1, 1].bar(range(len(pair_ids)), correlation_values)
    axes[1, 1].set_title('Correlation Coefficient')
    axes[1, 1].set_xlabel('Image Pairs')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_xticks(range(len(pair_ids)))
    axes[1, 1].set_xticklabels(pair_ids, rotation=45)
    axes[1, 1].set_ylim([0, 1])
    
    # 6. Difference distribution histogram
    all_diffs = mean_diff_values + max_diff_values
    axes[1, 2].hist(all_diffs, bins=20, alpha=0.7, label='Difference Distribution')
    axes[1, 2].set_title('Difference Value Distribution')
    axes[1, 2].set_xlabel('Difference Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'summary_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save numerical statistics
    with open(output_path / 'statistics.txt', 'w', encoding='utf-8') as f:
        f.write("Registration Results Difference Statistical Report\n")
        f.write("=" * 50 + "\n\n")
        
        for stats in all_stats:
            f.write(f"Image Pair: {stats['pair_id']}\n")
            f.write(f"  Mean Squared Error (MSE): {stats['mse']:.2f}\n")
            f.write(f"  Peak Signal-to-Noise Ratio (PSNR): {stats['psnr']:.2f} dB\n")
            f.write(f"  Mean Intensity Difference: {stats['mean_diff']:.2f}\n")
            f.write(f"  Max Intensity Difference: {stats['max_diff']:.2f}\n")
            f.write(f"  Correlation Coefficient: {stats['correlation']:.4f}\n")
            f.write("-" * 30 + "\n")
        
        # Overall statistics
        f.write("\nOverall Statistics:\n")
        f.write(f"  Average MSE: {np.mean(mse_values):.2f}\n")
        f.write(f"  Average PSNR: {np.mean(psnr_values):.2f} dB\n") if psnr_values else None
        f.write(f"  Average Mean Difference: {np.mean(mean_diff_values):.2f}\n")
        f.write(f"  Average Correlation: {np.mean(correlation_values):.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Registration Results Difference Visualization Tool')
    parser.add_argument('--input_dir', type=str, default='stage2_results_test',
                        help='Input directory containing registration results')
    parser.add_argument('--output_dir', type=str, default='registration_comparison',
                        help='Output directory for comparison visualizations')
    
    args = parser.parse_args()
    
    print("Starting registration results difference analysis...")
    analyze_all_pairs(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
