"""PatchEmb: patchEmb."""

from torch import nn


class PatchEmb(nn.Module):
    def __init__(self, img_size=224, p_size=16, in_chans=3, emb_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.p_size = p_size
        self.num_pat = (img_size // p_size) * (img_size // p_size)

        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=p_size, stride=p_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x=self.proj(x)
        x=x.flatten(2).transpose(1, 2)  # [B, num_pat, emb_dim]
        return x
