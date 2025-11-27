from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

dtype = torch.cuda.FloatTensor
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def input_coord_3d(image_depth, image_height, image_width):
    """Generate meshgrid for a 3D coordinate in [-1, 1]

    Return: 2D tensor with shape [D*H*W, 3]
    """
    tensors = [
        torch.linspace(-1, 1, steps=image_depth),
        torch.linspace(-1, 1, steps=image_height),
        torch.linspace(-1, 1, steps=image_width),
    ]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = rearrange(mgrid, "h w d c -> (h w d) c")  # [h*w*d, 3]

    return mgrid


def input_space_3d(image_depth, image_height, image_width):
    """Generate 3D grid coordinate in [-1, 1]

    Return: 4D tensor of shape [D, H, W, 3]
    """
    tensors = [
        torch.linspace(-1, 1, steps=image_depth),
        torch.linspace(-1, 1, steps=image_height),
        torch.linspace(-1, 1, steps=image_width),
    ]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)  # [d, h, w, 3]

    return mgrid


def space2coord(space_grid):
    """Convert space meshgrid to coordinates"""
    return rearrange(space_grid, "d h w c -> (d h w) c")


class SphericalEmbedding(nn.Module):
    """
    Spherical encoding method of NeRF.

    This class defines a function that projects (x, y, z) to r = |x * ri|,
    where |ri| = 1, and then embeds r to (r, sin(2^k r), cos(2^k r), ...).
    """

    def __init__(self, theta_degree, phi_degree, L, logscale=True, device=DEVICE):
        """
        Initializes the SphericalEmbedding module.

        Args:
            theta_degree (float): The degree step size for theta.
            phi_degree (float): The degree step size for phi.
            L (float): The encoding length.
            logscale (bool, optional): Whether to use logscale for frequency bands. Defaults to True.
        """
        super(SphericalEmbedding, self).__init__()
        theta = torch.arange(0, 180, theta_degree) * np.pi / 180  # [theta_num]
        phi = torch.arange(phi_degree, 180, phi_degree) * np.pi / 180  # [phi_num]
        encoding_length = int(np.floor(L))
        theta_num = len(theta)
        phi_num = len(phi)
        self.out_channels = (theta_num * phi_num + 1) * (2 * encoding_length)
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, L - 1, encoding_length)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (L - 1), encoding_length)

        x = torch.matmul(
            torch.sin(phi).unsqueeze(-1), torch.cos(theta).unsqueeze(-1).T
        )  # [phi_num, theta_num]
        y = torch.matmul(
            torch.sin(phi).unsqueeze(-1), torch.sin(theta).unsqueeze(-1).T
        )  # [phi_num, theta_num]
        z = torch.cos(phi).unsqueeze(-1)  # [phi_num, 1]
        z = z.repeat(1, x.shape[-1])  # [phi_num, theta_num]
        fourier_mapping = torch.stack((z, x, y), axis=-1)  # [phi_num, theta_num, 3]
        fourier_mapping = (
            fourier_mapping.view(-1, fourier_mapping.shape[-1]).to(device).T
        )  # [phi_num*theta_num, 3]
        self.fourier_mapping = torch.concat(
            [fourier_mapping, torch.tensor([[1], [0], [0]]).to(device)], axis=1
        )

    def forward(self, node):
        """
        Performs forward pass through the SphericalEmbedding module.

        Args:
            node (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        radial_proj = torch.matmul(node, self.fourier_mapping)
        out = []
        for l in self.freq_bands:
            for func in self.funcs:
                out += [func(l * np.pi * radial_proj)]
        return torch.cat(out, -1)


class PhysicsInformedEllipsoidalEmbedding(nn.Module):
    """
    Physics-Informed Ellipsoidal Encoding (PIEE) method

    The class defines an encoding that projects 3D coordinates $x = (x1, x2, x3)$
    to $r = |x * ri|$, and then embeds r within the optical system's OTF cutoff
    frequence range.
    """

    def __init__(
        self,
        theta_degree,
        phi_degree,
        L,
        fz_cut,
        fy_cut,
        fx_cut,
        logscale=True,
        freq_logscale=False,
        device=DEVICE,
    ):
        super(PhysicsInformedEllipsoidalEmbedding, self).__init__()
        self.theta = torch.arange(0, 180, theta_degree) * np.pi / 180
        self.phi = torch.arange(phi_degree, 180, 90 - phi_degree) * np.pi / 180
        self.encoding_length = int(np.floor(L))
        self.theta_num = len(self.theta)
        self.phi_num = len(self.phi)
        self.out_channels = (self.theta_num * self.phi_num + 1) * (2 * self.encoding_length)
        self.fz_cut = fz_cut
        self.fy_cut = fy_cut
        self.fx_cut = fx_cut
        self.funcs = [torch.sin, torch.cos]
        self.construct_fourier_mapping(logscale, freq_logscale, device)
        return None

    def construct_fourier_mapping(self, logscale, freq_logscale, device):
        # determine the fourier mapping matrix to be the from of:
        # 2^(l/(L-1))*[sin\phi*cos\theta*f_{rc}, sin\phi*sin\theta*f_{rc}, cos\phi*f_{rz}]
        # or to be the form of:
        # 2^(1/(L-1))*[sin\phi*cos\theta*(2f_{rc})^l, sin\phi*sin\theta*(2f_{rc})^l, cos\phi*(2f_{rz})^l]
        phi = self.phi
        theta = self.theta
        x = torch.matmul(torch.sin(phi).unsqueeze(-1), torch.cos(theta).unsqueeze(-1).T)
        y = torch.matmul(torch.sin(phi).unsqueeze(-1), torch.sin(theta).unsqueeze(-1).T)
        z = torch.cos(phi).unsqueeze(-1).repeat(1, x.shape[-1])
        fourier_mapping = torch.stack((z, x, y), axis=-1)
        fourier_mapping = fourier_mapping.view(-1, fourier_mapping.shape[-1]).to(device).T
        fourier_mapping = torch.concat(
            [fourier_mapping, torch.tensor([[1], [0], [0]]).to(device)], axis=1
        )  # [3, phi_num*theta_num]

        def exp2_matrix(fz_cut, fy_cut, fx_cut, L, l, freq_logscale):
            if freq_logscale:
                construct_element = lambda fc, l: np.power(fc, l / (L - 1))
            else:
                construct_element = lambda fc, l: np.power(2.0, l - (L - 1)) * fc

            z_element = construct_element(fz_cut, l).astype(np.float32)
            y_element = construct_element(fy_cut, l).astype(np.float32)
            x_element = construct_element(fx_cut, l).astype(np.float32)
            return torch.diag(torch.tensor([z_element, y_element, x_element]))

        fourier_mapping_list = []
        for l in range(self.encoding_length):
            exp_matrix = exp2_matrix(
                self.fz_cut,
                self.fy_cut,
                self.fx_cut,
                self.encoding_length,
                l,
                freq_logscale,
            ).to(device)
            fourier_mapping_list.append(torch.matmul(exp_matrix, fourier_mapping))
        self.fourier_mapping_unfolded = torch.cat(fourier_mapping_list, axis=1)
        return None

    def forward(self, node):
        radial_proj = torch.matmul(node, self.fourier_mapping_unfolded)
        # out = [node] # maybe help for low frequency component
        out = []
        for func in self.funcs:
            out += [func(np.pi * radial_proj)]
        return torch.cat(out, -1)


class RadialCartesianEmbedding(nn.Module):
    """
    Radial Cartesian Embedding module.

    Args:
        dia_digree (int): The degree of discretization for theta values.
        L_xy (int): The number of frequency bands for xy coordinates.
        L_z (int): The number of frequency bands for z coordinate.
        logscale (bool, optional): Whether to use logarithmic scale for frequency bands. Defaults to True.
    """

    def __init__(self, dia_digree, L_xy, L_z, logscale=True, device=DEVICE):
        super(RadialCartesianEmbedding, self).__init__()
        self.dia_digree = dia_digree
        theta = torch.arange(0, 180, dia_digree) * np.pi / 180
        theta_num = len(theta)
        self.L_xy = L_xy
        self.L_z = L_z
        self.out_channels = 2 * L_xy * theta_num + 2 * L_z
        self.funcs = [torch.sin, torch.cos]

        s = torch.sin(theta).unsqueeze(-1)
        c = torch.cos(theta).unsqueeze(-1)
        self.fourier_mapping = torch.cat((s, c), axis=-1).to(device).T  # [2, theta_num]

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, L_xy - 1, L_xy)
            self.freq_bands_z = 2 ** torch.linspace(0, L_z - 1, L_z)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (L_xy - 1), L_xy)
            self.freq_bands_z = torch.linspace(1, 2 ** (L_z - 1), L_z)

    def forward(self, node):
        """
        Forward pass of the RadialCartesianEmbedding module.

        Args:
            node (torch.Tensor): Input tensor representing the node.

        Returns:
            torch.Tensor: Output tensor after applying the radial cartesian embedding.
        """
        xy_freq = torch.matmul(node[:, 1:], self.fourier_mapping)
        out = []
        for l in self.freq_bands:
            for func in self.funcs:
                out += [func(l * np.pi * xy_freq)]
        for l in self.freq_bands_z:
            for func in self.funcs:
                out += [func(l * np.pi * node[:, 0].unsqueeze(-1))]
        return torch.cat(out, -1)


class GaussianEmbedding(nn.Module):
    def __init__(self, in_channels, scale, out_channels, device=DEVICE):
        super(GaussianEmbedding, self).__init__()
        self.scale = scale
        self.out_channels = out_channels * 2
        self.funcs = [torch.sin, torch.cos]

        np.random.seed(10)
        fourier_mapping = np.random.normal(0, self.scale, (in_channels, out_channels))
        self.fourier_mapping = torch.tensor(fourier_mapping).float().to(device)

        return None

    def forward(self, in_node):
        """
        Forward pass of the GaussianEmbedding module.

        Args:
            node (torch.Tensor): Input tensor representing the node.

        Returns:
            torch.Tensor: Output tensor after applying the Gaussian embedding.
        """
        radial_proj = torch.matmul(in_node, self.fourier_mapping)
        out = []
        for func in self.funcs:
            out += [func(radial_proj)]
        return torch.cat(out, -1)


class Embedding(nn.Module):
    """Encoding method of NeRF"""

    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


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

        return obj


class Seminerf(nn.Module):  # Spherical encoded fluorescence density field
    def __init__(
        self,
        D=6,
        W=128,
        skips=[4],
        out_channels=1,
        encoding_mode="PIEE",
        radial_encoding_angle=9,
        radial_encoding_depth=6,
        cartesian_encoding_depth=7,
        zenith_encoding_angle=45,
        gaussian_scale=26,
        gaussian_num=256,
        freq_logscale=False,
        phys_params={},
        device=DEVICE,
    ):
        super(Seminerf, self).__init__()
        if encoding_mode == "spherical":
            self.embedding = SphericalEmbedding(
                radial_encoding_angle,
                zenith_encoding_angle,
                radial_encoding_depth,
                device=device,
            )
        elif encoding_mode == "radial_cartesian":
            self.embedding = RadialCartesianEmbedding(
                radial_encoding_angle,
                radial_encoding_depth,
                cartesian_encoding_depth,
                device=device,
            )
        elif encoding_mode == "cartesian":
            self.embedding = Embedding(3, cartesian_encoding_depth)
        elif encoding_mode == "gaussian":
            self.embedding = GaussianEmbedding(3, gaussian_scale, gaussian_num, device)
        elif encoding_mode == "PIEE":
            if len(phys_params.keys()) == 0:
                phys_params = {
                    "fz_cut": 2 ** (radial_encoding_depth - 1),
                    "fy_cut": 2 ** (radial_encoding_depth - 1),
                    "fx_cut": 2 ** (radial_encoding_depth - 1),
                }
            self.embedding = PhysicsInformedEllipsoidalEmbedding(
                radial_encoding_angle,
                zenith_encoding_angle,
                radial_encoding_depth,
                device=device,
                freq_logscale=freq_logscale,
                **phys_params,
            )
        else:
            raise NotImplementedError(
                f"Encoding mode {encoding_mode} is not implemented for Seminerf."
            )
        in_channels = self.embedding.out_channels
        self.nerf = NeRF(D, W, skips, in_channels, out_channels)

    def forward(self, node):
        """
        Inputs:
            node: (B, 3) the position of the nodes
        Outputs:
            out: (B, 1), intensity
        """
        x = self.embedding(node)
        out = self.nerf(x)

        return out
