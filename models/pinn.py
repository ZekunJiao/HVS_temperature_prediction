import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNNet(nn.Module):
    def __init__(self, layers):
        """
        Initializes the PINN neural network.
        Args:
            layers (list of int): List defining the network structure.
                                  e.g., [input_dim, hidden1, hidden2, ..., output_dim]
                                  input_dim should be 3 (for t, x, y).
                                  output_dim should be 1 (for T).
        """
        super(PINNNet, self).__init__()
        module_list = []
        for i in range(len(layers) - 1):
            module_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                module_list.append(nn.Tanh())  # Activation function
        self.net = nn.Sequential(*module_list)

    def forward(self, txy_normalized):
        """
        Forward pass of the network.
        Args:
            txy_normalized (torch.Tensor): Input tensor of shape (N, 3) for normalized (t, x, y) coordinates.
        Returns:
            torch.Tensor: Predicted temperature T, shape (N, 1).
        """
        return self.net(txy_normalized)

def pde_loss_heat_equation_normalized_inputs(
    txy_collocation_normalized,  # Normalized [0,1] coordinates (t_n, x_n, y_n). Shape (N_coll, 3)
    model_net,                   # PINNNet, expects normalized inputs
    D_map,                       # Physical D_map (Ny, Nx)
    dx, dy,                      # Physical grid spacing of D_map
    coord_scales                 # Tuple/list [t_scale, x_scale, y_scale] (e.g., t_max, x_max, y_max)
                                 # Used to scale derivatives and coords for D_map sampling.
):
    """
    Computes the PDE loss for the heat equation: T_t = D * (T_xx + T_yy).
    Assumes PINN inputs are normalized, and scales derivatives to physical units.
    D_map is sampled using physical coordinates derived from normalized inputs and scales.
    """
    txy_collocation_normalized.requires_grad_(True)
    T_pred_collocation = model_net(txy_collocation_normalized)  # T = f(t_n, x_n, y_n)

    t_scale, x_scale, y_scale = coord_scales[0], coord_scales[1], coord_scales[2]

    # Gradients w.r.t. normalized coordinates (t_n, x_n, y_n)
    grad_T_norm = torch.autograd.grad(T_pred_collocation, txy_collocation_normalized,
                                      grad_outputs=torch.ones_like(T_pred_collocation),
                                      create_graph=True)[0]
    T_t_norm = grad_T_norm[:, 0:1]
    T_x_norm = grad_T_norm[:, 1:2]
    T_y_norm = grad_T_norm[:, 2:3]

    # Physical first derivatives using chain rule: dT/dt = (dT/dt_n) * (1/t_scale)
    T_t = T_t_norm / t_scale if t_scale > 1e-9 else T_t_norm
    T_x = T_x_norm / x_scale if x_scale > 1e-9 else T_x_norm
    T_y = T_y_norm / y_scale if y_scale > 1e-9 else T_y_norm

    # Second derivatives (w.r.t. normalized coordinates first)
    grad_Tx_norm = torch.autograd.grad(T_x_norm, txy_collocation_normalized,
                                       grad_outputs=torch.ones_like(T_x_norm),
                                       create_graph=True)[0]
    T_xx_norm = grad_Tx_norm[:, 1:2]  # d(T_x_norm)/dx_n

    grad_Ty_norm = torch.autograd.grad(T_y_norm, txy_collocation_normalized,
                                       grad_outputs=torch.ones_like(T_y_norm),
                                       create_graph=True)[0]
    T_yy_norm = grad_Ty_norm[:, 2:3]  # d(T_y_norm)/dy_n

    # Physical second derivatives: d2T/dx2 = T_xx_norm / x_scale^2
    T_xx = T_xx_norm / (x_scale**2) if x_scale > 1e-9 else T_xx_norm
    T_yy = T_yy_norm / (y_scale**2) if y_scale > 1e-9 else T_yy_norm

    # Interpolate D_map using PHYSICAL coordinates derived from normalized inputs
    ny_grid, nx_grid = D_map.shape
    
    # txy_collocation_normalized has (t_n, x_n, y_n), assumed to be in [0,1]
    x_coords_norm_in = txy_collocation_normalized[:, 1:2]  # x_n
    y_coords_norm_in = txy_collocation_normalized[:, 2:3]  # y_n

    # Convert normalized inputs [0,1] to physical coordinates for D_map sampling
    x_coords_phy = x_coords_norm_in * x_scale
    y_coords_phy = y_coords_norm_in * y_scale
    
    # Normalize physical coordinates to [-1,1] for F.grid_sample's requirements
    # max_x_phy_on_D_grid = (nx_grid - 1) * dx
    # max_y_phy_on_D_grid = (ny_grid - 1) * dy
    # gs_x_norm maps x_coords_phy to [-1,1] over the D_map's physical extent
    
    # Max extent of D_map's grid
    D_grid_max_x = (nx_grid - 1) * dx if nx_grid > 1 else dx 
    D_grid_max_y = (ny_grid - 1) * dy if ny_grid > 1 else dy

    gs_x_norm = (x_coords_phy / D_grid_max_x) * 2.0 - 1.0 if D_grid_max_x > 1e-9 else torch.zeros_like(x_coords_phy)
    gs_y_norm = (y_coords_phy / D_grid_max_y) * 2.0 - 1.0 if D_grid_max_y > 1e-9 else torch.zeros_like(y_coords_phy)
    
    if nx_grid == 1: gs_x_norm = torch.zeros_like(x_coords_phy) # Center on the single column
    if ny_grid == 1: gs_y_norm = torch.zeros_like(y_coords_phy) # Center on the single row

    gs_x_norm = torch.clamp(gs_x_norm, -1.0, 1.0)
    gs_y_norm = torch.clamp(gs_y_norm, -1.0, 1.0)
    
    D_map_reshaped = D_map.unsqueeze(0).unsqueeze(0)  # (1, 1, Ny, Nx)
    num_collocation_pts = txy_collocation_normalized.shape[0]
    
    # grid_sample expects (x,y) where x indexes width, y indexes height.
    sample_grid = torch.stack((gs_x_norm.squeeze(-1), gs_y_norm.squeeze(-1)), dim=-1)  # (N_coll, 2)
    # Reshape for grid_sample: (N_batch_input, H_out, W_out, 2_coords)
    # We want N_coll output points, treat them as H_out=N_coll, W_out=1
    sample_grid = sample_grid.reshape(1, num_collocation_pts, 1, 2)

    D_at_collocation = F.grid_sample(D_map_reshaped, sample_grid,
                                     mode='bilinear', padding_mode='border', align_corners=True)
    # Output D_at_collocation: (1, 1, N_coll, 1). Reshape to (N_coll, 1).
    D_at_collocation = D_at_collocation.reshape(num_collocation_pts, 1)

    # PDE residual: T_t - D * (T_xx + T_yy)
    pde_residual = T_t - D_at_collocation * (T_xx + T_yy)
    loss_pde = torch.mean(pde_residual**2)
    
    return loss_pde
