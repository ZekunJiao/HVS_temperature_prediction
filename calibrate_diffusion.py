import torch
import torch.nn.functional as F
from utils import rk4_step
import re
import matplotlib.pyplot as plt
from datetime import datetime

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
        
        if loss.item() < 1e-5 and epoch > 10: # ensure some epochs run
            print(f"Stopping early at epoch {epoch+1} due to low loss.")
            break
        
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


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    simulation_file_name = "0514_202854_simulation_n10_nx100_ny100_dt5e-05_dmin0.1_dmax0.3_nblobs20_radius30.pt"

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

    d_regs_est_numpy, training_losses, D_pred_final_tensor = calibrate_diffusivity(
        T_obs_full, dx=0.01, dy=0.01, dt=dt_val, 
        D=D_ground_truth, # Used only if run_per_gridpoint_mode is False
        d_init=0.15, 
        epochs=100,  
        lr=0.01,     
        device=device,
        calibrate_per_gridpoint=run_per_gridpoint_mode
    )

    if run_per_gridpoint_mode:
        print(f"Estimated d_regs (per-gridpoint mode stats): min={d_regs_est_numpy.min():.4f}, max={d_regs_est_numpy.max():.4f}, mean={d_regs_est_numpy.mean():.4f}")
    else:
        print("Estimated d_regs (per-region mode):", [f'{x:.4f}' for x in d_regs_est_numpy])
    
    print("Ground truth unique D values from source file:", torch.unique(D_ground_truth).cpu().tolist())

    # Visualize the training losses
    plt.figure(figsize=(12, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    mode_string = 'Per-Gridpoint' if run_per_gridpoint_mode else 'Per-Region'
    plt.title(f'Training Loss (Mode: {mode_string})')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f'./result/diffusion_calibration/Loss_{current_time}_{simulation_file_name}_{mode_string}.png')
    plt.close() # Close loss plot figure

    # Visualize D_ground_truth and D_pred_final
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    mode_string_for_suptitle = 'Per-Gridpoint' if run_per_gridpoint_mode else 'Per-Region'
    fig.suptitle(f'Diffusivity Maps (Mode: {mode_string_for_suptitle})')

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
    plt.savefig(f'./result/diffusion_calibration/D_Maps_{current_time}_{simulation_file_name}_{mode_string_for_filename}.png')
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
