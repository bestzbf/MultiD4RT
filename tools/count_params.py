"""Count parameters for D4RT base / large / giant variants."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.d4rt import create_d4rt


def fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def count(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    variants = ['base', 'large', 'giant']
    print(f"{'variant':<8} {'encoder':<14} {'decoder':<14} {'total':<14} {'trainable':<14}")
    print("-" * 70)
    for v in variants:
        model = create_d4rt(
            variant=v,
            encoder_pretrained=False,
            use_videomae=False,
            use_videomae_v2=False,
            use_vggt=False,
            use_monst3r=False,
            use_timm=False,
            use_videomae_backbone=False,
            use_videomae_init=False,
            use_timm_init=False,
        )
        enc_total, _ = count(model.encoder)
        dec_total, _ = count(model.decoder)
        total, trainable = count(model)
        print(f"{v:<8} {fmt(enc_total):<14} {fmt(dec_total):<14} {fmt(total):<14} {fmt(trainable):<14}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':
    main()
