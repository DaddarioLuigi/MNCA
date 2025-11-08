import torch
import torch.nn.functional as F
import torch.nn as nn
from .MixtureNCANoise import MixtureNCANoise

class ExtendedMixtureNCANoise(MixtureNCANoise):
    """
    Extended MixtureNCANoise that inherits from MixtureNCANoise base class.
    Adds support for configurable neighborhood sizes beyond the standard 3x3
    without rewriting the original mixture/forward logic.
    """
    def __init__(self, update_nets, num_rules=5, state_dim=16, hidden_dim=128,
                 dropout=0, temperature=1.0, device="cuda", num_latent_dims=1,
                 use_alive_mask=True, alive_threshold=0.1, alive_channel=3,
                 maintain_seed=True, residual=True, grid_type="square",
                 modality="image", filter_type="sobel",
                 save_internal_noise=False, distribution=None, seed_value=1.0,
                 neighborhood_size=3):
        # Call parent constructor to keep original behavior
        super(ExtendedMixtureNCANoise, self).__init__(
            update_nets=update_nets,
            num_rules=num_rules,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature,
            device=device,
            num_latent_dims=num_latent_dims,
            use_alive_mask=use_alive_mask,
            alive_threshold=alive_threshold,
            alive_channel=alive_channel,
            maintain_seed=maintain_seed,
            residual=residual,
            grid_type=grid_type,
            modality=modality,
            filter_type=filter_type,
            save_internal_noise=save_internal_noise,
            distribution=distribution,
            seed_value=seed_value
        )

        # Store neighborhood size and override perception filters if needed
        self.neighborhood_size = neighborhood_size
        if neighborhood_size > 3:
            self._setup_extended_perception_filters(device)

        # Re-bind perceive after (re)creating kernels
        if modality == "image":
            self.perceive = self._perceive_image
        elif modality == "tensor":
            self.perceive = self._perceive_tensor

    def _setup_extended_perception_filters(self, device):
        """Setup perception filters for extended neighborhood sizes"""
        kernel_size = self.neighborhood_size
        padding = kernel_size // 2

        # Identity filter (center pixel)
        identity = torch.zeros(kernel_size, kernel_size, dtype=torch.float32).to(device)
        identity[padding, padding] = 1.0

        if self.filter_type == "sobel":
            # Extended Sobel filters
            if kernel_size == 3:
                sobel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).to(device)
                sobel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).to(device)
            elif kernel_size == 4:
                # Extended 4x4 Sobel
                sobel_x = torch.tensor([[-1, -1, 1, 1],
                                       [-2, -2, 2, 2],
                                       [-2, -2, 2, 2],
                                       [-1, -1, 1, 1]], dtype=torch.float32).to(device)
                sobel_y = torch.tensor([[-1, -2, -2, -1],
                                       [-1, -2, -2, -1],
                                       [1, 2, 2, 1],
                                       [1, 2, 2, 1]], dtype=torch.float32).to(device)
            elif kernel_size == 5:
                sobel_x = torch.tensor([[-1, -2, 0, 2, 1],
                                       [-2, -4, 0, 4, 2],
                                       [-1, -2, 0, 2, 1],
                                       [-1, -2, 0, 2, 1],
                                       [-1, -2, 0, 2, 1]], dtype=torch.float32).to(device)
                sobel_y = torch.tensor([[-1, -2, -1, -1, -1],
                                       [-2, -4, -2, -2, -2],
                                       [0, 0, 0, 0, 0],
                                       [2, 4, 2, 2, 2],
                                       [1, 2, 1, 1, 1]], dtype=torch.float32).to(device)
            elif kernel_size == 6:
                # Extended 6x6 Sobel
                sobel_x = torch.tensor([[-1, -2, -2, 2, 2, 1],
                                       [-2, -4, -4, 4, 4, 2],
                                       [-2, -4, -4, 4, 4, 2],
                                       [-2, -4, -4, 4, 4, 2],
                                       [-1, -2, -2, 2, 2, 1],
                                       [-1, -2, -2, 2, 2, 1]], dtype=torch.float32).to(device)
                sobel_y = torch.tensor([[-1, -2, -2, -2, -1, -1],
                                       [-2, -4, -4, -4, -2, -2],
                                       [-2, -4, -4, -4, -2, -2],
                                       [2, 4, 4, 4, 2, 2],
                                       [2, 4, 4, 4, 2, 2],
                                       [1, 2, 2, 2, 1, 1]], dtype=torch.float32).to(device)
            elif kernel_size == 7:
                sobel_x = torch.tensor([[-1, -2, -3, 0, 3, 2, 1],
                                       [-2, -4, -6, 0, 6, 4, 2],
                                       [-3, -6, -9, 0, 9, 6, 3],
                                       [-1, -2, -3, 0, 3, 2, 1],
                                       [-1, -2, -3, 0, 3, 2, 1],
                                       [-1, -2, -3, 0, 3, 2, 1],
                                       [-1, -2, -3, 0, 3, 2, 1]], dtype=torch.float32).to(device)
                sobel_y = torch.tensor([[-1, -2, -3, -1, -1, -1, -1],
                                       [-2, -4, -6, -2, -2, -2, -2],
                                       [-3, -6, -9, -3, -3, -3, -3],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [3, 6, 9, 3, 3, 3, 3],
                                       [2, 4, 6, 2, 2, 2, 2],
                                       [1, 2, 3, 1, 1, 1, 1]], dtype=torch.float32).to(device)
            else:
                raise ValueError(f"Unsupported kernel size for sobel: {kernel_size}. Supported: 3, 4, 5, 6, 7")

            self.register_buffer('sobel_x_kernel',
                sobel_x.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
            self.register_buffer('sobel_y_kernel',
                sobel_y.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))

        else:  # laplacian
            if kernel_size == 3:
                laplacian = torch.tensor([[1, 1, 1],
                                          [1, -8, 1],
                                          [1, 1, 1]], dtype=torch.float32).to(device)
            elif kernel_size == 4:
                # Extended 4x4 Laplacian
                laplacian = torch.tensor([[1, 1, 1, 1],
                                          [1, 1, 1, 1],
                                          [1, 1, -12, 1],
                                          [1, 1, 1, 1]], dtype=torch.float32).to(device)
            elif kernel_size == 5:
                laplacian = torch.tensor([[1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1],
                                          [1, 1, -24, 1, 1],
                                          [1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1]], dtype=torch.float32).to(device)
            elif kernel_size == 6:
                # Extended 6x6 Laplacian
                laplacian = torch.tensor([[1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, -32, 1, 1],
                                          [1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1]], dtype=torch.float32).to(device)
            elif kernel_size == 7:
                laplacian = torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, -48, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32).to(device)
            else:
                raise ValueError(f"Unsupported kernel size for laplacian: {kernel_size}. Supported: 3, 4, 5, 6, 7")

            self.register_buffer('laplacian_kernel',
                laplacian.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))

        self.register_buffer('identity_kernel',
            identity.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))

    def _perceive_image(self, x):
        padding = self.neighborhood_size // 2
        identity = F.conv2d(x, self.identity_kernel, padding=padding, groups=self.state_dim)

        if self.filter_type == "sobel":
            sobel_x = F.conv2d(x, self.sobel_x_kernel, padding=padding, groups=self.state_dim)
            sobel_y = F.conv2d(x, self.sobel_y_kernel, padding=padding, groups=self.state_dim)
            return torch.cat([identity, sobel_x, sobel_y], dim=1)
        else:  # laplacian
            laplacian = F.conv2d(x, self.laplacian_kernel, padding=padding, groups=self.state_dim)
            return torch.cat([identity, laplacian], dim=1)

    def _perceive_tensor(self, x):
        identity = F.conv2d(x, self.identity_kernel, padding=0, groups=self.state_dim)

        if self.filter_type == "sobel":
            sobel_x = F.conv2d(x, self.sobel_x_kernel, padding=0, groups=self.state_dim)
            sobel_y = F.conv2d(x, self.sobel_y_kernel, padding=0, groups=self.state_dim)
            return torch.cat([identity, sobel_x, sobel_y], dim=1)
        else:  # laplacian
            laplacian = F.conv2d(x, self.laplacian_kernel, padding=0, groups=self.state_dim)
            return torch.cat([identity, laplacian], dim=1)


