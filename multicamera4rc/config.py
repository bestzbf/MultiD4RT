"""Architecture settings; the giant preset matches the released 4RC modules."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MultiCamera4RCConfig:
    embed_dim: int = 1536
    encoder_depth: int = 40
    encoder_heads: int = 24
    out_layers: tuple[int, ...] = (19, 27, 33, 39)
    alt_start: int = 13
    positional_image_size: int = 518
    patch_size: int = 14
    ffn_layer: str = "swiglufused"
    motion_depth: int = 4
    motion_heads: int = 16
    dpt_features: int = 256
    dpt_channels: tuple[int, ...] = (256, 512, 1024, 1024)
    decoder_memory: str = "all_observations"
    time_origin: str = "clip_start"
    geometry_chunk_size: int | None = 8
    attention_chunk_size: int | None = 128
    visibility_chunk_size: int | None = 4096
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.decoder_memory not in ("all_observations", "target_time"):
            raise ValueError("decoder_memory must be all_observations or target_time")
        if self.time_origin not in ("clip_start", "absolute"):
            raise ValueError("time_origin must be clip_start or absolute")
        if self.embed_dim <= 0 or self.embed_dim % 2:
            raise ValueError("embed_dim must be a positive even integer")
        if self.encoder_heads <= 0 or self.motion_heads <= 0:
            raise ValueError("attention head counts must be positive")
        for heads in (self.encoder_heads, self.motion_heads):
            if self.embed_dim % heads or (self.embed_dim // heads) % 4:
                raise ValueError("2D RoPE requires head dimensions divisible by four")
        if len(self.out_layers) != 4 or len(self.dpt_channels) != 4:
            raise ValueError("both DPT heads require exactly four encoder taps")
        if tuple(sorted(set(self.out_layers))) != tuple(self.out_layers):
            raise ValueError("out_layers must be distinct and increasing")
        if not 0 <= self.alt_start < self.encoder_depth:
            raise ValueError("alt_start must lie inside the encoder")
        if any(x < self.alt_start or x >= self.encoder_depth or x % 2 == 0
               for x in self.out_layers):
            raise ValueError("all taps must be global blocks at/after alt_start")
        if self.alt_start % 2 == 0:
            raise ValueError("the first fusion block must be a global (odd-indexed) block")
        if min(self.patch_size, self.motion_depth, self.dpt_features) <= 0:
            raise ValueError("patch, depth and feature settings must be positive")
        for chunk_size in (self.attention_chunk_size, self.visibility_chunk_size, self.geometry_chunk_size):
            if chunk_size is not None and (isinstance(chunk_size, bool)
                    or not isinstance(chunk_size, int) or chunk_size <= 0):
                raise ValueError("chunk settings must be positive integers or None for whole queries")
        if self.dpt_features % 2 or any(c <= 0 for c in self.dpt_channels):
            raise ValueError("DPT features must be even and channels positive")

    @classmethod
    def tiny(cls, **overrides):
        """Same complete architecture, reduced widths for CPU correctness tests."""
        values = dict(embed_dim=64, encoder_depth=8, encoder_heads=4,
                      out_layers=(1, 3, 5, 7), alt_start=1,
                      positional_image_size=56, ffn_layer="mlp", motion_heads=4,
                      dpt_features=32, dpt_channels=(32, 64, 64, 64))
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key in ("out_layers", "dpt_channels"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self):
        return asdict(self)
