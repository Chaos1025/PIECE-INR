# Author: Chenyu Xu
# Last Date: 2023/08/23
# Version: 1.1
# Description: Implement those INR network architectures like NeRF, SIREN, WIRE, WIRE2d, GAUSS,
#       and their corresponding layer modules.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random


class none_func(nn.Module):
    """Just return the input tensor and do nothing"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


act_funcs = {
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(True),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
    "none": nn.Identity(),
}


# sub-modules
class SineLayer(nn.Module):
    """A linear layer using the 'sin' function as unlinear activation.

    If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the nonlinearity.
    Different signals may require different omega_0 in the first layer - this is a hyperparameter.

    If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class GaussLayer(nn.Module):
    """Drop in replacement for SineLayer but with Gaussian non linearity"""

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30, scale=10.0
    ):
        """
        is_first, and omega_0 are not used.
        """
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale = scale
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return torch.exp(-((self.scale * self.linear(input)) ** 2))


class ComplexGaborLayer(nn.Module):
    """Implicit representation with complex Gabor nonlinearity

    Inputs:
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega0: Frequency of Gabor sinusoid term
        sigma0: Scaling of Gabor Gaussian term
        trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=40.0,
        trainable=True,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class ComplexGaborLayer2D(nn.Module):
    """Implicit representation with complex Gabor nonlinearity with 2D activation function

    Inputs:
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega0: Frequency of Gabor sinusoid term
        sigma0: Scaling of Gabor Gaussian term
        trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=10.0,
        trainable=True,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        # Second Gaussian window
        self.scale_orth = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)

        scale_x = lin
        scale_y = self.scale_orth(input)

        freq_term = torch.exp(1j * self.omega_0 * lin)

        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0 * self.scale_0 * arg)

        return freq_term * gauss_term


# main modules
class MLP(nn.Module):
    """MLP network with NeRF-like architecture."""

    def __init__(
        self,
        in_channels=84,
        out_channels=1,
        hidden_features=256,
        hidden_layers=8,
        outermost_linear=True,
        out_activation="sigmoid",
        skips=[2, 4, 6],
    ):
        super(MLP, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.in_channels = in_channels
        self.skips = skips

        # hidden layers
        for i in range(hidden_layers):
            if i == 0:
                layer = nn.Linear(in_channels, hidden_features)
            elif i in skips:
                layer = nn.Linear(hidden_features + in_channels, hidden_features)
            else:
                layer = nn.Linear(hidden_features, hidden_features)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"uvxy_encoding_{i+1}", layer)

        # output layers
        if out_activation not in act_funcs.keys():
            raise ValueError(f"out_activation must be one of {list(act_funcs.keys())}")

        self.rgb = nn.Sequential(
            nn.Linear(hidden_features, out_channels), act_funcs[out_activation]
        )

    def forward(self, x):

        input_uvxy = x
        uvxy_ = input_uvxy
        for i in range(self.hidden_layers):
            if i in self.skips:
                uvxy_ = torch.cat([input_uvxy, uvxy_], -1)
            uvxy_ = getattr(self, f"uvxy_encoding_{i+1}")(uvxy_)

        out = self.rgb(uvxy_)
        return out


# NeRF network code from the paper of CoCoA
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, skips=[4], in_channels=63, out_channels=180):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        # self.beta = beta
        # self.max_val = max_val

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.enc_last = nn.Linear(W, W)

        # output layers
        self.post = nn.Sequential(
            nn.Linear(W, W // 2), nn.ReLU(True), nn.Linear(W // 2, out_channels)
        )

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        enc_last = self.enc_last(xyz_)
        obj = self.post(enc_last)

        # obj = nn.Softplus(beta = self.beta)(obj) if self.beta is not None else self.max_val * nn.Sigmoid()(obj)

        return obj


class SIREN(nn.Module):
    """
    The SIREN network model, using `sin` function as activation function.
    Param `outermost_linear` determine whether the output layer should be linear layer or still be sine layer.
    Param `out_activation` determines to use `sigmoid` or `ReLU` function as the last layer's activation func.
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        hidden_features=256,
        hidden_layers=1,
        outermost_linear=True,
        out_activation="sigmoid",
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()
        self.net = []
        self.net.append(
            SineLayer(in_channels, hidden_features, is_first=True, omega_0=first_omega_0)
        )
        for i in range(hidden_layers):
            self.net.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0)
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_channels)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(hidden_features, out_channels, is_first=False, omega_0=hidden_omega_0)
            )

        if out_activation in act_funcs.keys():
            self.net.append(act_funcs[out_activation])
        elif out_activation != None:
            raise ValueError(f"out_activation must be one of {list(act_funcs.keys())}")
        else:
            pass

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        # torch.clip_(output, min=0.0, max=1.0)
        return output


class Gauss(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        hidden_features=256,
        hidden_layers=2,
        outermost_linear=True,
        out_activation="sigmoid",
        first_omega_0=30,
        hidden_omega_0=30.0,
        scale=10.0,
    ):
        super().__init__()
        self.nonlin = GaussLayer

        self.net = []
        self.net.append(
            self.nonlin(
                in_channels, hidden_features, is_first=True, omega_0=first_omega_0, scale=scale
            )
        )
        for i in range(hidden_layers):
            self.net.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    scale=scale,
                )
            )
        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_channels))
        else:
            self.net.append(
                self.nonlin(
                    hidden_features,
                    out_channels,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    scale=scale,
                )
            )

        if out_activation in act_funcs.keys():
            self.net.append(act_funcs[out_activation])
        elif out_activation != None:
            raise ValueError(f"out_activation must be one of {list(act_funcs.keys())}")
        else:
            pass

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


class WIRE(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        hidden_features=256,
        hidden_layers=1,
        outermost_linear=True,
        out_activation="sigmoid",
        first_omega_0=30,
        hidden_omega_0=30.0,
        scale=50.0,
    ):
        super().__init__()

        self.nonlin = ComplexGaborLayer
        hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.out_activation = out_activation

        self.net = []
        self.net.append(
            self.nonlin(
                in_channels, hidden_features, omega0=first_omega_0, sigma0=scale, is_first=True
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                self.nonlin(hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=scale)
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_channels, dtype=dtype)
            self.net.append(final_linear)
        else:
            self.net.append(
                self.nonlin(hidden_features, out_channels, omega0=hidden_omega_0, sigma0=scale)
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        if self.out_activation == None:
            return output.real
        elif self.out_activation in act_funcs.keys():
            return act_funcs[self.out_activation](output.real)
        else:
            raise ValueError(f"out_activation must be one of {list(act_funcs.keys())}")


class WIRE2D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        hidden_features=256,
        hidden_layers=1,
        outermost_linear=True,
        out_activation="sigmoid",
        first_omega_0=30,
        hidden_omega_0=30.0,
        scale=10.0,
    ):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer2D
        self.out_activation = out_activation

        # Since complex numbers are two real numbers, reduce the number of
        # hidden parameters by 4
        hidden_features = int(hidden_features / 2)
        dtype = torch.cfloat
        self.wavelet = "gabor"

        # Legacy parameter

        self.net = []
        self.net.append(
            self.nonlin(
                in_channels, hidden_features, omega0=first_omega_0, sigma0=scale, is_first=True
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                self.nonlin(hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=scale)
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_channels, dtype=dtype)
            self.net.append(final_linear)
        else:
            self.net.append(
                self.nonlin(hidden_features, out_channels, omega0=hidden_omega_0, sigma0=scale)
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)

        if self.wavelet == "gabor":
            output = output.real

        if self.out_activation == None:
            return output
        elif self.out_activation in act_funcs.keys():
            return act_funcs[self.out_activation](output)
        else:
            raise ValueError(f"out_activation must be one of {list(act_funcs.keys())}")


class HashGridNeRF(nn.Module):
    def __init__(
        self, hash_grid_config, nerf_hid_dim=128, nerf_hid_layer_num=5, dtype=None
    ) -> None:
        import tinycudann as tcnn

        super().__init__()

        self.hash_grid_config = hash_grid_config
        self.mlp_config = (
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": nerf_hid_dim,
                "n_hidden_layers": nerf_hid_layer_num,
            }
            if nerf_hid_dim <= 128
            else {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": nerf_hid_dim,
                "n_hidden_layers": nerf_hid_layer_num,
            }
        )

        self.dtype = dtype
        self.hash_grid_encoding = tcnn.Encoding(
            3, self.hash_grid_config, dtype=dtype, seed=random.randint(0, 1524367)
        )
        self.mlp = tcnn.Network(
            self.hash_grid_encoding.n_output_dims,
            1,
            self.mlp_config,
            seed=random.randint(0, 1524367),
        )

    def forward(self, coords):
        orig_shape = coords.shape

        encoded_pos = self.hash_grid_encoding(coords.reshape(-1, 3))

        density = self.mlp(encoded_pos).reshape(*orig_shape[:-1], -1)

        density = density.float()

        return density
