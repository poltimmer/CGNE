import torch
from torch import nn, Tensor
from torch.nn import functional as F

from CGNE.utils.misc import round_hidden_dim


def get_dims(in_dim, out_dim, hidden_dim=128, n_layers=7):
    """
    Returns a list of dimensions for a linear network with `n_layers` layers.
    The layer dims interpolate from in_dim, to hidden_dim in the middle, to out_dim.
    The interpolation is linear, but the hidden dims should always be either
    a power of 2, or a multiple of 32 (or 64 when > 256).
    """
    if n_layers < 2:
        raise ValueError("num_layers must be at least 2")

    # Calculate half the number of layers
    half_layers = (n_layers - 2) // 2

    # Generate dimensions from `in_dim` to `hidden_dim`
    increase_dims = [int(in_dim + i * (hidden_dim - in_dim) / (half_layers + 1)) for i in range(1, half_layers + 1)]
    increase_dims = [round_hidden_dim(dim) for dim in increase_dims]

    # If num_layers is odd, we add a middle layer
    middle_layers = [hidden_dim] if n_layers % 2 == 0 else [hidden_dim, hidden_dim]

    # Generate dimensions from `hidden_dim` to `out_dim`
    decrease_dims = [int(hidden_dim - i * (hidden_dim - out_dim) / (half_layers + 1)) for i in
                     range(1, half_layers + 1)]
    decrease_dims = [round_hidden_dim(dim) for dim in decrease_dims]

    return [in_dim] + increase_dims + middle_layers + decrease_dims + [out_dim]


def linear(in_dim, out_dim, hidden_dim=128, n_layers=7, activation=nn.GELU):
    """
    Gives a Sequential of linear layers.
    The layer dims interpolate from in_dim, to hidden_dim in the middle, to out_dim.
    The interpolation is linear, but the hidden dims should always be either
    a power of 2, or a multiple of 32 (or 64 when > 256).
    """
    dims = get_dims(in_dim, out_dim, hidden_dim, n_layers)

    # Construct the layers based on the calculated dims
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # Avoid activation after the last linear layer
            layers.append(activation())

    return nn.Sequential(*layers)


def inject(x: Tensor, y: Tensor, method: str = 'cat'):
    if method not in ['cat', 'add', 'mul']:
        raise ValueError(f"inject_method must be 'cat', 'add', or 'mul', but got {method}")
    if x.dim() != y.dim():
        shape = [x.shape[0]] + [-1] + [1] * (x.dim() - 2)
        y = y.view(shape)
    if x.dim() == 4 and x.shape[-2:] != y.shape[-2:]:
        y = F.interpolate(y, size=x.shape[-2:], mode='nearest-exact')
    if method == 'cat':
        return torch.cat([x, y], dim=1)
    else:
        # Match channels
        factor = x.shape[1] // y.shape[1]
        remainder = x.shape[1] % y.shape[1]
        y = torch.cat([y] * factor + [y[:, :remainder]], dim=1)
        if method == 'add':
            return x + y
        elif method == 'mul':
            return x * (1 + y)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_1x1_conv=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
        )
        if use_1x1_conv:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        x = self.block(x) + self.skip(x)
        return F.gelu(x)


class ResBlockUp(ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsampling=True, use_1x1_conv=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, use_1x1_conv)
        if upsampling:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        return super().forward(x)


class ResBlockDown(ResBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsampling=True, use_1x1_conv=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, use_1x1_conv)
        if downsampling:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        x = self.downsample(x)
        return super().forward(x)


class SelfAttention(nn.Module):
    """
    From https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py
    """

    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    """
    From https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    From https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py
    """

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """
    From https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x=None):
        x = self.up(x)
        if skip_x is not None:
            x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x


class BorderMask(nn.Module):
    """
    Returns a mask of the border when given a tensor of binary attachment images.
    """

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(data=[[1,  1, 0],
                                    [1, -6, 1],
                                    [0,  1, 1]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x):
        mask = self.conv((x[:, :1] > 0.5).type(torch.float32))
        return mask > 0
