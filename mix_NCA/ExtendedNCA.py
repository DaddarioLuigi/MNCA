import torch
import torch.nn.functional as F
import torch.nn as nn
from .NCA import NCA

class ExtendedNCA(NCA):
    """
    Extended NCA that inherits from NCA base class
    Adds support for configurable neighborhood sizes beyond the standard 3x3
    
    Supports different neighborhood configurations:
    - 3x3 (8 neighbors) - standard (uses parent class)
    - 4x4 (12 neighbors) - extended
    - 5x5 (24 neighbors) - extended
    - 6x6 (32 neighbors) - very extended
    - 7x7 (48 neighbors) - very extended
    - Custom configurations
    """
    def __init__(self, update_net, state_dim=16, hidden_dim=128, dropout=0, 
                 device="cuda", use_alive_mask=True, alive_threshold=0.1, 
                 alive_channel=3, maintain_seed=True, residual=True, 
                 grid_type="square", modality="image", filter_type="sobel",
                 distribution=None, random_updates=False, seed_value=1.0,
                 neighborhood_size=3):
        
        # Call parent constructor, but we'll override perception filters
        super(ExtendedNCA, self).__init__(
            update_net=update_net,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device,
            use_alive_mask=use_alive_mask,
            alive_threshold=alive_threshold,
            alive_channel=alive_channel,
            maintain_seed=maintain_seed,
            residual=residual,
            grid_type=grid_type,
            modality=modality,
            filter_type=filter_type,
            distribution=distribution,
            random_updates=random_updates,
            seed_value=seed_value
        )
        
        # Add neighborhood size parameter
        self.neighborhood_size = neighborhood_size
        
        # Override perception filters if neighborhood size > 3
        if neighborhood_size > 3:
            self._setup_extended_perception_filters(device)
        
        # Re-set perception function after filter setup
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
            sobel_x = torch.zeros(kernel_size, kernel_size, dtype=torch.float32).to(device)
            sobel_y = torch.zeros(kernel_size, kernel_size, dtype=torch.float32).to(device)
            
            if kernel_size == 3:
                # Standard 3x3 Sobel
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
                # Extended 5x5 Sobel
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
                # Extended 7x7 Sobel
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
            
            self.register_buffer('sobel_x_kernel', 
                sobel_x.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
            self.register_buffer('sobel_y_kernel', 
                sobel_y.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
                
        else:  # laplacian
            # Extended Laplacian filters
            laplacian = torch.zeros(kernel_size, kernel_size, dtype=torch.float32).to(device)
            
            if kernel_size == 3:
                # Standard 3x3 Laplacian
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
                # Extended 5x5 Laplacian
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
                # Extended 7x7 Laplacian
                laplacian = torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, -48, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32).to(device)
            
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
            return torch.cat([identity, laplacian, laplacian], dim=1)  # Duplicate to maintain same dimensions

    def _perceive_tensor(self, x):
        identity = F.conv2d(x, self.identity_kernel, padding=0, groups=self.state_dim)
        
        if self.filter_type == "sobel":
            sobel_x = F.conv2d(x, self.sobel_x_kernel, padding=0, groups=self.state_dim)
            sobel_y = F.conv2d(x, self.sobel_y_kernel, padding=0, groups=self.state_dim)
            return torch.cat([identity, sobel_x, sobel_y], dim=1)
        else:  # laplacian
            laplacian = F.conv2d(x, self.laplacian_kernel, padding=0, groups=self.state_dim)
            return torch.cat([identity, laplacian, laplacian], dim=1)  # Duplicate to maintain same dimensions

    def forward(self, x, num_steps, seed_loc=None, return_history=False):
        frames = []
        
        for i in range(num_steps):
            if torch.isnan(x).any():
                print(f"NaN detected in state at step {i}")
                break
                
            # Get pre-update alive mask if using
            if self.use_alive_mask:
                alive_mask_pre = nn.functional.max_pool2d(
                    x[:, self.alive_channel:self.alive_channel+1], 
                    3, stride=1, padding=1
                ) > self.alive_threshold
            
            # Create update mask based on modality
            if self.residual:
                if self.modality == "image":
                    update_mask = torch.rand(*x.shape, device=x.device) > self.dropout
                elif self.modality == "tensor":
                    update_mask = torch.rand(*x[:,:,1:2,1:2].shape, device=x.device) > self.dropout
            
            # Perceive neighborhood and compute update
            perceived = self.perceive(x)
            dx = self.update(perceived)
            
            # Transform network output based on distribution
            if self.distribution == "normal":
                # Split output into mean and log_std
                mean, log_std = torch.chunk(dx, 2, dim=1)
                log_std = torch.clamp(log_std, min=-6, max=6)
                # Ensure positive std dev
                std = torch.clamp(torch.nn.functional.softplus(log_std), min=1e-6, max=10) * 0.06
                # Sample from normal distribution
                if self.random_updates:
                    dx = mean + std * torch.randn_like(mean)
                else:
                    dx = mean
            
            # Apply update mask
            if self.residual:
                if self.modality == "image":
                    dx = dx * update_mask.float()
                elif self.modality == "tensor":
                    dx = dx * update_mask.float()
            
            # Apply alive mask
            if self.use_alive_mask:
                dx = dx * alive_mask_pre.float()
            
            # Maintain seed if specified
            if self.maintain_seed and seed_loc is not None:
                dx[:, :, seed_loc[0], seed_loc[1]] = 0
            
            # Update state
            x = x + dx
            
            # Maintain seed value
            if self.maintain_seed and seed_loc is not None:
                x[:, :, seed_loc[0], seed_loc[1]] = self.seed_value
            
            if return_history:
                frames.append(x.clone())
        
        if return_history:
            return x, frames
        return x
