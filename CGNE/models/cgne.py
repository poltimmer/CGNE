import torch
from torch import nn
from torch.nn import functional as F

from CGNE.models.layers import linear, ResBlockDown, ResBlock, ResBlockUp, get_dims, inject
from CGNE.utils.misc import round_hidden_dim


class ConvEncoder(nn.Module):
    """
    Encodes x (64x64) to z (z_dim), or to spatial z (z_dim, 8, 8)
    """

    def __init__(self, c_in=1, z_dim=32, hidden_dim=32, spatial_z=False, down=ResBlockDown):
        super().__init__()
        self.down1 = down(c_in, hidden_dim)
        self.down2 = down(hidden_dim, hidden_dim * 2)
        self.down3 = down(hidden_dim * 2, hidden_dim * 4)
        self.spatial_z = spatial_z
        if spatial_z:
            self.conv = ResBlock(hidden_dim * 4, hidden_dim * 4)
            self.outc = nn.Conv2d(hidden_dim * 4, z_dim * 2, kernel_size=3, padding=1)
        else:
            self.down4 = down(hidden_dim * 4, hidden_dim * 4)
            self.pool = nn.AdaptiveAvgPool2d(2)
            self.fc = nn.Linear(hidden_dim * 4 * 2 * 2, z_dim * 2)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        if self.spatial_z:
            x = self.conv(x)
            x = self.outc(x)
        else:
            x = self.down4(x)
            x = self.pool(x)
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
        z_loc, z_scale = x.chunk(2, dim=1)
        z_scale = F.softplus(z_scale)
        z_scale += 1e-7  # To prevent nans
        return z_loc, z_scale


class ConditionalEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32, cond_dim=7, z_dim=32, spatial_z=False, cond_embed_dim=-1,
                 cond_hidden_dim=-1, cond_n_layers=-1):
        super().__init__()
        if cond_embed_dim > 0:
            if cond_hidden_dim == -1:
                cond_hidden_dim = round_hidden_dim((cond_dim + cond_embed_dim) / 2)
            self.y_embed = linear(cond_dim, cond_embed_dim, hidden_dim=cond_hidden_dim, n_layers=cond_n_layers)
        else:
            cond_embed_dim = cond_dim
            self.y_embed = nn.Identity()

        self.encoder = ConvEncoder(c_in=in_channels + cond_embed_dim, z_dim=z_dim, hidden_dim=hidden_dim,
                                   spatial_z=spatial_z)

    def forward(self, x, y):
        y_embed = self.y_embed(y)
        y_projection = y_embed.view(y_embed.shape[0], -1, 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
        x = torch.cat((x, y_projection), dim=1)
        return self.encoder(x)


class ConditionalPrior(ConditionalEncoder):
    def __init__(self, x_channels=1, hidden_dim=32, y_dim=7, z_dim=32, spatial_z=False, y_embed_dim=-1, y_hidden_dim=-1,
                 y_n_layers=-1):
        super().__init__(in_channels=x_channels, hidden_dim=hidden_dim, cond_dim=y_dim, z_dim=z_dim,
                         spatial_z=spatial_z, cond_embed_dim=y_embed_dim, cond_hidden_dim=y_hidden_dim,
                         cond_n_layers=y_n_layers)


class ApproximatePosterior(nn.Module):
    def __init__(self, x_channels=1, hidden_dim=32, y_dim=7, z_dim=32, spatial_z=False, y_embed_dim=-1, y_hidden_dim=-1,
                 y_n_layers=-1):
        super().__init__()
        self.enc = ConditionalEncoder(in_channels=x_channels * 2, hidden_dim=hidden_dim, cond_dim=y_dim, z_dim=z_dim,
                                      spatial_z=spatial_z, cond_embed_dim=y_embed_dim, cond_hidden_dim=y_hidden_dim,
                                      cond_n_layers=y_n_layers)

    def forward(self, x_t1, x_t, y):
        return self.enc(torch.cat((x_t1, x_t), dim=1), y)


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, z_dim=32, hidden_dim=64,
                 spatial_z=False, z_emb_dim=-1, z_emb_hidden_dim=-1, z_emb_n_layers=-1, z_inject_method='cat'):
        super().__init__()
        self.spatial_z = spatial_z
        if z_inject_method not in ['cat', 'add', 'mul']:
            raise ValueError(f"z_inject_method must be 'cat', 'add', or 'mul', but got {z_inject_method}")
        self.z_inject_method = z_inject_method
        if z_emb_dim > 0:
            if z_emb_hidden_dim == -1:
                z_emb_hidden_dim = round_hidden_dim((z_dim + z_emb_dim) / 2)
            if spatial_z:
                conv_layers = []
                conv_dims = get_dims(z_dim, z_emb_dim, hidden_dim, z_emb_n_layers)
                for i in range(len(conv_dims) - 1):
                    conv_layers.append(ResBlock(conv_dims[i], conv_dims[i + 1]))
                    if i < len(conv_dims) - 2:  # Avoid activation after the last linear layer
                        conv_layers.append(nn.GELU())
                self.z_embed = nn.Sequential(*conv_layers)
            else:
                self.z_embed = linear(z_dim, z_emb_dim, hidden_dim=z_emb_hidden_dim, n_layers=z_emb_n_layers)
            z_dim = z_emb_dim
        else:
            self.z_embed = nn.Identity()
        if z_inject_method != 'cat':
            z_dim = 0
        self.inc = ResBlock(c_in, hidden_dim)
        self.down1 = ResBlockDown(hidden_dim + z_dim, hidden_dim * 2)
        self.down2 = ResBlockDown(hidden_dim * 2 + z_dim, hidden_dim * 4)
        self.down3 = ResBlockDown(hidden_dim * 4 + z_dim, hidden_dim * 4)
        self.up1 = ResBlockUp(hidden_dim * 4 + z_dim, hidden_dim * 4)
        self.up2 = ResBlockUp(hidden_dim * 4 * 2 + z_dim, hidden_dim * 2)
        self.up3 = ResBlockUp(hidden_dim * 2 * 2 + z_dim, hidden_dim)
        self.full_conv = ResBlock(hidden_dim * 2 + z_dim, hidden_dim)
        self.outc = nn.Conv2d(hidden_dim, c_out, kernel_size=3, padding=1)

    def forward(self, x, z):
        z = self.z_embed(z)
        x1 = self.inc(x)
        x1 = inject(x1, z, method=self.z_inject_method)
        x2 = self.down1(x1)
        x2 = inject(x2, z, method=self.z_inject_method)
        x3 = self.down2(x2)
        x3 = inject(x3, z, method=self.z_inject_method)
        x4 = self.down3(x3)
        x4 = inject(x4, z, method=self.z_inject_method)
        x = self.up1(x4)
        x = self.up2(torch.cat([x, x3], dim=1))
        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.full_conv(torch.cat([x, x1], dim=1))
        output = self.outc(x)
        return output


class SpatialZDecoder(nn.Module):
    def __init__(self, z_dim=32, out_dim=7, fc_hidden_dim=512, fc_n_layers=5):
        super().__init__()
        self.inc = ResBlock(z_dim, z_dim * 2)
        self.down1 = ResBlockDown(z_dim * 2, z_dim * 4)
        self.down2 = ResBlockDown(z_dim * 4, z_dim * 4)
        self.fc = linear(z_dim * 4 * 2 * 2, out_dim, hidden_dim=fc_hidden_dim, n_layers=fc_n_layers)

    def forward(self, z):
        x = self.inc(z)
        x = self.down1(x)
        x = self.down2(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)
