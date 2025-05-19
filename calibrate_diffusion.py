import torch
import torch.nn.functional as F
from utils import rk4_step
import re
import matplotlib.pyplot as plt
from datetime import datetime
from pinn import PINNNet, pde_loss_heat_equation_normalized_inputs

def calibrate_diffusivity(
    T_obs,           # Tensor of shape (nt, ny, nx): observed temperature data
    dx, dy, dt,      # Spatial and temporal step sizes
    D,               # full diffusivity map (ny, nx) with discrete blob regions (used if not calibrate_per_gridpoint)
    d_init=0.2,      # Initial guess for region diffusivities or per-gridpoint diffusivities
    lr=1e-2,         # Learning rate
    epochs=200,      # Number of optimization steps
    device='cpu',    # 'cpu' or 'cuda'
    calibrate_per_gridpoint=False # New flag to control calibration mode
):
    # Move data to device
    T_obs = T_obs[0].to(device)
    nt, ny, nx = T_obs.shape

    if calibrate_per_gridpoint:
        print("Calibrating diffusivity per gridpoint.")
        # Initialize d_regs as a full (ny, nx) map of learnable parameters
        init_map = torch.full((ny, nx), d_init, device=device)
        d_regs = torch.nn.Parameter(init_map)
        # D (ground truth) is not used for defining parameters in this mode
    else:
        print("Calibrating diffusivity per region.")
        unique_vals = torch.unique(D) # D is the ground truth map, used to define regions
        print("ground truth unique D values (for region definition):", unique_vals)
        if unique_vals.numel() == 0:
            raise ValueError("Input D for region-based calibration is empty or has no unique values.")
        n_regions = unique_vals.numel()
        masks = [(D == v).to(device) for v in unique_vals]
        
        init_vals = torch.full((n_regions,), d_init, device=device)
        d_regs = torch.nn.Parameter(init_vals)

    optimizer = torch.optim.Adam([d_regs], lr=lr)
    losses = []

    # Calibration loop
    for epoch in range(epochs):
        if calibrate_per_gridpoint:
            D_pred_epoch = d_regs # d_regs is the (ny, nx) map directly
        else:
            D_pred_epoch = torch.zeros((ny, nx), device=device)
            for i in range(n_regions): # n_regions and masks are defined in the 'else' block above
                D_pred_epoch = D_pred_epoch + d_regs[i] * masks[i]

        # Forward RK4 simulation without in-place writes
        Ts = [T_obs[0]]
        for t_idx in range(nt - 1):
            Ts.append(rk4_step(Ts[-1], D_pred_epoch, dx, dy, dt))

        # Compute weighted MSE loss
        weights = torch.linspace(0.1, 1.0, nt, device=device)
        loss = sum(weights[t_idx] * F.mse_loss(Ts[t_idx], T_obs[t_idx]) for t_idx in range(nt))

        # Backprop and update region diffusivities
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Enforce positivity
        d_regs.data.clamp_(min=1e-6)

        losses.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs -1: # Print more often or for specific epochs
            loss_item = loss.item()
            if calibrate_per_gridpoint:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_item:.6f}, d_regs stats: (min: {d_regs.min().item():.4f}, max: {d_regs.max().item():.4f}, mean: {d_regs.mean().item():.4f})")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_item:.6f}, estimates: {[f'{x:.4f}' for x in d_regs.data.tolist()]}")
        
        # if loss.item() < 5e-6 and epoch > 10: # ensure some epochs run
        #     print(f"Stopping early at epoch {epoch+1} due to low loss.")
        #     break

    # Reconstruct final D_pred with optimized d_regs for returning (and plotting)
    if calibrate_per_gridpoint:
        final_D_pred = d_regs.detach() # d_regs is already the full map
    else:
        final_D_pred = torch.zeros((ny, nx), device=device)
        for i in range(n_regions): # n_regions and masks from 'else' block for region-based
            final_D_pred = final_D_pred + d_regs[i] * masks[i]

    # d_regs contains the optimized parameters (either per-region or per-gridpoint)
    # final_D_pred is always the full (ny, nx) map based on optimized d_regs
    return d_regs.detach().cpu().numpy(), losses, final_D_pred


def calibrate_diffusivity_pinn(
    T_obs,           # Tensor of shape (nt, ny, nx): observed temperature data from a single simulation frame
    dx, dy, dt,      # Spatial and temporal step sizes from the simulation
    D_gt_for_regions,# Ground truth D map (ny, nx) used for defining regions if not in per-gridpoint mode.
                     # Should be the D map from which T_obs was generated or a similar structural map.
    d_init=0.1,      # Initial guess for region diffusivities or per-gridpoint diffusivities
    lr=1e-3,         # Learning rate for Adam optimizer
    epochs=1000,     # Number of optimization steps
    device='cpu',    # 'cpu' or 'cuda'
    calibrate_per_gridpoint=False, # True for per-gridpoint D, False for per-region D
    pinn_layers=[3, 64, 64, 64, 1], # PINN architecture: [input_dim (t,x,y), ..., output_dim (T)]
    n_collocation_points=10000,    # Number of collocation points for PDE loss
    lambda_pde=1.0                 # Weight for the PDE loss component
):
    """
    Calibrates diffusivity D(x,y) using a Physics-Informed Neural Network (PINN).
    The PINN learns T(t,x,y) and the diffusivity parameters are optimized concurrently.
    """
    T_obs_device = T_obs.to(device) # T_obs is expected as (nt, ny, nx)
    _nt, ny, nx = T_obs_device.shape

    print(f"Starting PINN-based diffusivity calibration. Mode: {'Per-Gridpoint' if calibrate_per_gridpoint else 'Per-Region'}")
    print(f"T_obs shape: ({_nt}, {ny}, {nx}), dt: {dt}, dx: {dx}, dy: {dy}")

    # 1. Initialize Diffusivity Parameters (d_regs)
    if calibrate_per_gridpoint:
        print(f"Initializing d_regs as a ({ny}, {nx}) map with init value {d_init}.")
        init_map = torch.full((ny, nx), d_init, device=device, dtype=torch.float32)
        d_regs = torch.nn.Parameter(init_map)
        masks = None # Not used in per-gridpoint mode for D_pred construction here
        n_regions = None
    else:
        print(f"Initializing d_regs for regions based on D_gt_for_regions.")
        unique_vals = torch.unique(D_gt_for_regions)
        if unique_vals.numel() == 0:
            raise ValueError("Input D_gt_for_regions for region-based calibration is empty or has no unique values.")
        n_regions = unique_vals.numel()
        masks = [(D_gt_for_regions == v).to(device) for v in unique_vals]
        print(f"Found {n_regions} unique regions from D_gt_for_regions.")
        init_vals = torch.full((n_regions,), d_init, device=device, dtype=torch.float32)
        d_regs = torch.nn.Parameter(init_vals)

    # 2. Initialize PINNNet
    pinn_net = PINNNet(layers=pinn_layers).to(device)
    print(f"Initialized PINNNet with layers: {pinn_layers}")

    # 3. Optimizer
    optimizer = torch.optim.Adam(list(pinn_net.parameters()) + [d_regs], lr=lr)
    
    losses_total_history = []
    losses_data_history = []
    losses_pde_history = []

    # 4. Prepare Data for Data Loss Term & Coordinate Scaling
    # Physical coordinates for T_obs grid
    t_coords_phy = torch.linspace(0, (_nt - 1) * dt, _nt, device=device)
    y_coords_phy = torch.linspace(0, (ny - 1) * dy, ny, device=device)
    x_coords_phy = torch.linspace(0, (nx - 1) * dx, nx, device=device)

    # Determine scales for normalization (max physical coordinate values)
    # Handle single-point dimensions carefully: if nt=1, t_max is 0. Scale should be 1 to avoid div by zero in norm.
    t_max_phy = (_nt - 1) * dt if _nt > 1 else dt # if _nt=1, max_t is effectively dt or 0 if dt=0
    y_max_phy = (ny - 1) * dy if ny > 1 else dy
    x_max_phy = (nx - 1) * dx if nx > 1 else dx

    t_scale = t_max_phy if t_max_phy > 1e-9 else 1.0
    y_scale = y_max_phy if y_max_phy > 1e-9 else 1.0
    x_scale = x_max_phy if x_max_phy > 1e-9 else 1.0
    coord_scales = [t_scale, x_scale, y_scale] # Note: order (t,x,y) for consistency with PINN input
    print(f"Coordinate scales (t_max, x_max, y_max): {coord_scales}")

    # Create normalized grid for data loss input (PINN expects normalized input)
    T_grid_phy, X_grid_phy, Y_grid_phy = torch.meshgrid(t_coords_phy, x_coords_phy, y_coords_phy, indexing='ij')
    
    T_grid_norm = T_grid_phy / t_scale
    X_grid_norm = X_grid_phy / x_scale
    Y_grid_norm = Y_grid_phy / y_scale
    
    txy_data_normalized = torch.stack((T_grid_norm.flatten(), X_grid_norm.flatten(), Y_grid_norm.flatten()), dim=1)
    T_data_values = T_obs_device.flatten().unsqueeze(-1)
    print(f"Prepared normalized data input for PINN: {txy_data_normalized.shape}, Target T values: {T_data_values.shape}")

    # 5. Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Construct D_pred_epoch map from d_regs (current estimate of D(x,y))
        if calibrate_per_gridpoint:
            D_pred_epoch = d_regs
        else:
            D_pred_epoch = torch.zeros((ny, nx), device=device, dtype=torch.float32)
            for i in range(n_regions):
                D_pred_epoch = D_pred_epoch + d_regs[i] * masks[i]
        D_pred_epoch = D_pred_epoch.clamp(min=1e-7) # Ensure D is positive

        # a. Data Loss (using normalized coordinates as input to PINN)
        T_pred_data = pinn_net(txy_data_normalized)
        loss_data = F.mse_loss(T_pred_data, T_data_values)

        # b. PDE Loss
        # Generate NORMALIZE_PER_GRIDPOINTD collocation points [0,1] for (t_n, x_n, y_n)
        t_coll_norm = torch.rand(n_collocation_points, 1, device=device)  # Normalized t_n in [0,1]
        x_coll_norm = torch.rand(n_collocation_points, 1, device=device)  # Normalized x_n in [0,1]
        y_coll_norm = torch.rand(n_collocation_points, 1, device=device)  # Normalized y_n in [0,1]
        txy_collocation_normalized = torch.cat((t_coll_norm, x_coll_norm, y_coll_norm), dim=1)
        # `requires_grad` is set inside pde_loss_heat_equation_normalized_inputs

        loss_pde = pde_loss_heat_equation_normalized_inputs(
            txy_collocation_normalized, # Normalized inputs for PINN
            pinn_net,
            D_pred_epoch,             # Current physical D_map estimate
            dx, dy,                   # Physical grid spacing of D_map
            coord_scales              # Physical scales [t_max, x_max, y_max] for derivative scaling
        )

        # c. Total Loss
        total_loss = loss_data + lambda_pde * loss_pde

        # d. Backprop
        total_loss.backward()
        optimizer.step()

        # e. Enforce positivity of d_regs (diffusivity values)
        d_regs.data.clamp_(min=1e-7)

        losses_total_history.append(total_loss.item())
        losses_data_history.append(loss_data.item())
        losses_pde_history.append(loss_pde.item())

        if (epoch + 1) % 1 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss.item():.4e}, "
                  f"Data Loss: {loss_data.item():.4e}, PDE Loss: {loss_pde.item():.4e}")
            if calibrate_per_gridpoint:
                print(f"  d_regs stats: (min: {d_regs.min().item():.4f}, max: {d_regs.max().item():.4f}, "
                      f"mean: {d_regs.mean().item():.4f})")
            else:
                print(f"  d_regs estimates: {[f'{x:.4f}' for x in d_regs.data.tolist()]}")
    
    # Reconstruct final D_pred_map for returning
    final_D_pred_map = torch.zeros((ny, nx), device=device, dtype=torch.float32)
    if calibrate_per_gridpoint:
        final_D_pred_map = d_regs.detach().clone()
    else:
        for i in range(n_regions):
            final_D_pred_map = final_D_pred_map + d_regs[i].detach() * masks[i]
    
    # d_regs contains the optimized parameters (either per-region or per-gridpoint map)
    # losses_total_history contains the total loss history
    # final_D_pred_map is the full (ny, nx) map based on optimized d_regs
    return d_regs.detach().cpu().numpy(), (losses_total_history, losses_data_history, losses_pde_history), final_D_pred_map.cpu()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    simulation_file_name = "0517_131936_simulation_n10_nx100_ny100_dt5e-05_dmin0.1_dmax0.3_nblobs200_radius5.pt"

    data = torch.load(f'./datasets/simulation/{simulation_file_name}')
    T_obs_full = data['T'].to(device) # Keep original name for clarity before slicing
    D_ground_truth = data['D'].to(device)
    
    pattern = r"dt([0-9.eE+-]+)" # Simplified pattern for dt
    match = re.search(pattern, simulation_file_name)
    
    dt_val = 5e-5 # Default dt
    if match:
        dt_val = float(match.group(1))
        print(f"Extracted dt = {dt_val} from filename")
    else:
        print(f"Pattern for dt not found in filename. Using default dt={dt_val}.")
    
    # --- CHOOSE CALIBRATION MODE --- 
    run_per_gridpoint_mode = True # Set to True to calibrate each grid point, False for region-based
    # --------------------------------

    # --- CHOOSE CALIBRATION FUNCTION ---
    use_pinn_calibration = True # Set to True to use PINN calibration, False for original
    # ----------------------------------

    if use_pinn_calibration:
        print("\n--- Using PINN-based Calibration ---")
        # Ensure T_obs_full[0] is used, which is (nt, ny, nx)
        # D_ground_truth is used for region definition if not per-gridpoint
        d_params_pinn, losses_pinn, D_pred_pinn_tensor = calibrate_diffusivity_pinn(
            T_obs=T_obs_full[0], # Use the first frame: (nt, ny, nx)
            dx=0.01, dy=0.01, dt=dt_val,
            D_gt_for_regions=D_ground_truth, # Used only if run_per_gridpoint_mode is False
            d_init=0.15,
            lr=1e-4, # PINNs often require smaller LR for joint optimization
            epochs=20, # PINNs might need more epochs
            device=device,
            calibrate_per_gridpoint=run_per_gridpoint_mode,
            pinn_layers=[3, 40, 40, 40, 40, 1], # Example layers
            n_collocation_points=20000,
            lambda_pde=0.1 # Adjust PDE loss weight as needed
        )
        training_losses = losses_pinn[0] # Total loss for plotting
        D_pred_final_tensor = D_pred_pinn_tensor
        d_regs_est_numpy = d_params_pinn
        print("PINN calibration finished.")

    else:
        print("\n--- Using Original RK4-based Calibration ---")
        d_regs_est_numpy, training_losses, D_pred_final_tensor = calibrate_diffusivity(
            T_obs_full, dx=0.01, dy=0.01, dt=dt_val, 
            D=D_ground_truth, # Used only if run_per_gridpoint_mode is False
            d_init=0.15, 
            epochs=100,  
            lr=0.01,     
            device=device,
            calibrate_per_gridpoint=run_per_gridpoint_mode
        )
        print("Original calibration finished.")


    if run_per_gridpoint_mode:
        print(f"Estimated d_regs (per-gridpoint mode stats): min={d_regs_est_numpy.min():.4f}, max={d_regs_est_numpy.max():.4f}, mean={d_regs_est_numpy.mean():.4f}")
    else:
        print("Estimated d_regs (per-region mode):", [f'{x:.4f}' for x in d_regs_est_numpy])
    
    # Visualize the training losses
    plt.figure(figsize=(12, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    mode_string = 'Per-Gridpoint' if run_per_gridpoint_mode else 'Per-Region'
    calibration_method = "PINN" if use_pinn_calibration else "RK4"
    plt.title(f'Training Loss (Method: {calibration_method}, Mode: {mode_string})')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f'./result/diffusion_calibration/Loss_{current_time}_{calibration_method}_{mode_string}_{simulation_file_name}.png')
    plt.close() # Close loss plot figure

    # Visualize D_ground_truth and D_pred_final
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    mode_string_for_suptitle = 'Per-Gridpoint' if run_per_gridpoint_mode else 'Per-Region'
    calibration_method_title = "PINN" if use_pinn_calibration else "RK4"
    fig.suptitle(f'Diffusivity Maps (Method: {calibration_method_title}, Mode: {mode_string_for_suptitle}), Final Loss: {training_losses[-1]:.4e}')

    # Determine global min and max for consistent color scaling
    d_min = min(D_ground_truth.min(), D_pred_final_tensor.min()).item()
    d_max = max(D_ground_truth.max(), D_pred_final_tensor.max()).item()

    im1 = axs[0].imshow(D_ground_truth.cpu().numpy(), cmap='viridis', vmin=d_min, vmax=d_max)
    axs[0].set_title('Ground Truth D')
    axs[0].set_xlabel('nx')
    axs[0].set_ylabel('ny')
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    im2 = axs[1].imshow(D_pred_final_tensor.cpu().numpy(), cmap='viridis', vmin=d_min, vmax=d_max) # D_pred_final_tensor is already on device
    axs[1].set_title('Predicted D (Final)')
    axs[1].set_xlabel('nx')
    axs[1].set_ylabel('ny')
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    mode_string_for_filename = 'Grid' if run_per_gridpoint_mode else 'Region'
    calibration_method_filename = "PINN" if use_pinn_calibration else "RK4"
    plt.savefig(f'./result/diffusion_calibration/D_Maps_{current_time}_{calibration_method_filename}_{mode_string_for_filename}_{simulation_file_name}.png')
    plt.show()
    plt.close(fig) # Close D maps figure

    # For direct comparison of values, find unique values in D_pred_final
    unique_D_pred_tensor = torch.unique(D_pred_final_tensor)
    if run_per_gridpoint_mode:
        if len(unique_D_pred_tensor) > 30: # Arbitrary threshold for summary
            print(f"Unique values in final D_pred (per-gridpoint): {len(unique_D_pred_tensor)} distinct values. Min: {unique_D_pred_tensor.min().item():.4f}, Max: {unique_D_pred_tensor.max().item():.4f}, Mean: {unique_D_pred_tensor.float().mean().item():.4f}")
        else:
            print("Unique values in final D_pred (per-gridpoint, few unique):", [f'{x:.4f}' for x in unique_D_pred_tensor.cpu().tolist()])
    else:
        print("Unique values in final D_pred (per-region):", [f'{x:.4f}' for x in unique_D_pred_tensor.cpu().tolist()])
