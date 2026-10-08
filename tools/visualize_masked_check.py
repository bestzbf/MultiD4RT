#!/usr/bin/env python3
"""
Quick check to visualize BlendedMVS masked vs non-masked images
and verify dataset configuration is correct.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml

def main():
    # Load config
    with open(REPO_ROOT / 'configs/single_blendedmvs.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("BlendedMVS Dataset Configuration Check")
    print("=" * 60)
    print(f"\nConfig file: configs/single_blendedmvs.yaml")
    print(f"use_masked: {config['adapter_kwargs']['use_masked']}")
    print(f"root: {config['root']}")
    print()

    if not config['adapter_kwargs']['use_masked']:
        print("⚠️ WARNING: use_masked=false - will load unmasked images!")
        print("   Consider setting use_masked=true for better results.")
        return

    print("✅ use_masked=true - will load masked images (with distant areas removed)")
    print()

    # Create adapter
    from datasets.registry import create_adapter

    adapter = create_adapter(
        name=config['name'],
        root=config['root'],
        split='train',
        use_masked=True,  # Force true to test
        verbose=True
    )

    # Test loading a sample sequence
    seq_names = adapter.list_sequences()[:3]

    print("\n" + "=" * 60)
    print("Sample Sequence Analysis")
    print("=" * 60)

    n_seqs = len(seq_names)
    fig, axes = plt.subplots(n_seqs, 5, figsize=(20, n_seqs * 3))
    if n_seqs == 1:
        axes = [axes]

    for i, seq_name in enumerate(seq_names):
        seq_info = adapter.get_sequence_info(seq_name)

        clip = adapter.load_clip(seq_name, [0])
        img_0 = clip.images[0]  # RGB image
        depth_0 = clip.depths[0]  # Depth map

        print(f"\nSequence: {seq_name}")
        print(f"  Frames: {seq_info['num_frames']}, Size: {seq_info['width']}x{seq_info['height']}")
        print(f"  use_masked in metadata: {clip.metadata.get('use_masked', 'N/A')}")

        # Show RGB image
        axes[i][0].imshow(img_0.astype(np.uint8))
        axes[i][0].set_title('RGB Image (masked)' if clip.metadata.get('use_masked') else 'RGB Image (unmasked)')
        axes[i][0].axis('off')

        # Show depth (normalized)
        valid_depth = depth_0[depth_0 > 0]
        if len(valid_depth) > 0:
            depth_display = np.clip(depth_0 / max(valid_depth.max(), 0.01), 0, 1)
            min_d, max_d = valid_depth.min(), valid_depth.max()
        else:
            depth_display = np.zeros_like(depth_0)
            min_d, max_d = 0, 0

        axes[i][1].imshow(depth_display, cmap='viridis')
        axes[i][1].set_title(f'Depth (min={min_d:.3f}m, max={max_d:.3f}m)')
        axes[i][1].axis('off')

        # Count black pixels (masked regions)
        black_pixels = np.sum(np.all(img_0 == 0, axis=2))
        total_pixels = img_0.shape[0] * img_0.shape[1]
        mask_ratio = black_pixels / total_pixels * 100

        axes[i][2].imshow(np.zeros_like(img_0))
        axes[i][2].text(0.5, 0.5, f'Masked:\n{mask_ratio:.1f}%',
                       ha='center', va='center', fontsize=14,
                       color='white', weight='bold')
        axes[i][2].set_title('Mask Ratio (black pixels)')
        axes[i][2].axis('off')

        # Show histogram of depth values
        axes[i][3].hist(valid_depth, bins=100, alpha=0.7, edgecolor='black')
        if len(valid_depth) > 0:
            axes[i][3].axvline(min_d, color='r', linestyle='--', label=f'min={min_d:.3f}')
            axes[i][3].axvline(max_d, color='g', linestyle='--', label=f'max={max_d:.3f}')
        axes[i][3].set_xlabel('Depth (m)')
        axes[i][3].set_ylabel('Frequency')
        axes[i][3].set_title('Depth Histogram')
        if len(valid_depth) > 0:
            axes[i][3].legend(fontsize=8)
        axes[i][3].grid(True, alpha=0.3)

        # Summary text
        axes[i][4].text(0.1, 0.8,
                       f"Dataset: {clip.metadata.get('dataset_name')}\n"
                       f"Split: {clip.metadata.get('split')}\n"
                       f"use_masked: {clip.metadata.get('use_masked', False)}\n"
                       f"has_tracks: {clip.metadata.get('has_tracks', False)}\n"
                       f"valid pixels: {100-mask_ratio:.1f}%",
                       transform=axes[i][4].transAxes, fontsize=11,
                       family='monospace', verticalalignment='top')
        axes[i][4].axis('off')
        axes[i][4].set_xlim(0, 1)
        axes[i][4].set_ylim(0, 1)

    plt.tight_layout()
    output_file = 'outputs/blendedmvs_masked_visualization.png'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    print()

    # Summary statistics
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    all_mask_ratios = []
    for seq_name in seq_names:
        seq_info = adapter.get_sequence_info(seq_name)
        clip = adapter.load_clip(seq_name, [0])
        img_0 = clip.images[0]
        black_pixels = np.sum(np.all(img_0 == 0, axis=2))
        mask_ratio = black_pixels / (img_0.shape[0] * img_0.shape[1]) * 100
        all_mask_ratios.append(mask_ratio)
        print(f"  {seq_name}: {mask_ratio:.1f}% masked ({100-mask_ratio:.1f}% valid)")

    print(f"\nAverage masked ratio: {np.mean(all_mask_ratios):.2f}%")
    print(f"This means ~{100-np.mean(all_mask_ratios):.1f}% of pixels have valid depth supervision")
    print()

    if np.mean(all_mask_ratios) > 5:
        print("✅ Configuration is CORRECT! Using masked images reduces noise in training.")
    else:
        print("⚠️ Low mask ratio - verify masked images exist.")

if __name__ == '__main__':
    main()
