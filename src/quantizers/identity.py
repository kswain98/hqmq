from .base import KVQuantizer


class IdentityQuantizer(KVQuantizer):
    """fp16 passthrough — upper-bound baseline."""

    def quantize_K(self, k, layer_idx):
        return k

    def quantize_V(self, v, layer_idx):
        return v

    def bits_per_value(self):
        return 16.0
