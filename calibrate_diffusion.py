import torch
import torch.nn.functional as F
from utils import rk4_step
import re


def calibrate_diffusivity(
    T_obs,           # Tensor of shape (nt, ny, nx): observed temperature data
    dx, dy, dt,      # Spatial and temporal step sizes
    start_x, start_y, end_x, end_y,  # Region of 'd_in' in the domain
    d_in_init=0.5,   # Initial guess for inside-region diffusivity
    d_out_init=0.5,  # Initial guess for outside-region diffusivity
    lr=1e-2,         # Learning rate
    epochs=200,      # Number of optimization steps
    device='cpu'     # 'cpu' or 'cuda'
):
    # Move data to device
    T_obs = T_obs.to(device)
    print(T_obs.shape)
    T_obs = T_obs[0]
    nt, ny, nx = T_obs.shape
    
    # Initialize parameters (ensure they are within a positive range)
    d_in = torch.tensor(d_in_init, requires_grad=True, device=device)
    d_out = torch.tensor(d_out_init, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([d_in, d_out], lr=lr)
    losses = []
    
    # Calibration loop
    for epoch in range(epochs):
        # Build diffusivity map for this iteration
        D = torch.ones((ny, nx), device=device) * d_out
        D[start_y:end_y, start_x:end_x] = d_in
        
        # Simulate forward using RK4
        T = torch.empty_like(T_obs)
        T[0] = T_obs[0]  # Use observed initial condition
        for t in range(nt - 1):
            T[t + 1] = rk4_step(T[t], D, dx, dy, dt)
        
        weights = torch.linspace(0.1, 1.0, nt, device=device)   # ramp up weight
        loss = sum(weights[t] * F.mse_loss(T[t], T_obs[t]) for t in range(nt))
        
        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Keep parameters positive
        d_in.data.clamp_(1e-6, 10.0)
        d_out.data.clamp_(1e-6, 10.0)
        
        losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, d_in: {d_in.item():.4f}, d_out: {d_out.item():.4f}")
    
    return d_in.item(), d_out.item(), losses



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    simulation_file_name = "0514_122331_simulation_n10_t00.015_t0.015_nx100_ny100_dt5e-05_din0.1_dout0.3_sy44_ey80_sx23_ex39.pt"
    
    T_obs = torch.load(f'./datasets/simulation/{simulation_file_name}').to(device)

    pattern = r"dt([0-9.eE+-]+)_din(\d+\.\d+)_dout(\d+\.\d+)_sy(\d+)_ey(\d+)_sx(\d+)_ex(\d+)"
    match = re.search(pattern, simulation_file_name)

    if match:
        dt = float(match.group(1))
        din = float(match.group(2))
        dout = float(match.group(3))
        sy = int(match.group(4))
        ey = int(match.group(5))
        sx = int(match.group(6))
        ex = int(match.group(7))
        print(f"Extracted values:\ndt = {dt}\ndin = {din}\ndout = {dout}\nstart_y (sy) = {sy}\nend_y (ey) = {ey}\nstart_x (sx) = {sx}\nend_x (ex) = {ex}")
    else:
        print("Pattern not found in filename.")
    
    d_in_est, d_out_est, training_losses = calibrate_diffusivity(
        T_obs, dx=0.01, dy=0.01, dt=dt,
        start_x=sx, start_y=sy, end_x=ex, end_y=ey,
        device=device
    )

    print("Estimated d_in:", d_in_est, "Estimated d_out:", d_out_est)
