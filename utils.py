import torch

def gradient(T, dx, dy):
    """Gradient part of the PDE."""
    T_x = (torch.roll(T, shifts=-1, dims=0) - torch.roll(T, shifts=1, dims=0)) / (2 * dx)
    T_y = (torch.roll(T, shifts=-1, dims=1) - torch.roll(T, shifts=1, dims=1)) / (2 * dy)
    return T_x, T_y

def laplacian(T, dx, dy):
    """Laplacian part of the PDE."""
    T_xx = (torch.roll(T, shifts=-1, dims=0) - 2 * T + torch.roll(T, shifts=1, dims=0)) / dx**2
    T_yy = (torch.roll(T, shifts=-1, dims=1) - 2 * T + torch.roll(T, shifts=1, dims=1)) / dy**2

    return T_xx + T_yy

def heat_equation_rhs(T, D, dx, dy):
    """Compute the right-hand side of the heat equation."""
    T_x, T_y = gradient(T, dx, dy)
    D_x, D_y = gradient(D, dx, dy)
    
    diffusion_term = D * laplacian(T, dx, dy)
    gradient_term = D_x * T_x + D_y * T_y
    
    return diffusion_term + gradient_term

def rk4_step(T, D, dx, dy, dt):
    """Runge-Kutta 4 time stepping scheme for the heat equation."""
    k1 = heat_equation_rhs(T, D, dx, dy)
    k2 = heat_equation_rhs(T + 0.5 * dt * k1, D, dx, dy)
    k3 = heat_equation_rhs(T + 0.5 * dt * k2, D, dx, dy)
    k4 = heat_equation_rhs(T + dt * k3, D, dx, dy)
    return T + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def create_masked_input(full_field, observed_fraction=0.05):
    """Create a binary mask and the corresponding partial field, ensuring CUDA compatibility."""
    device = full_field.device  # Ensure operations are done on the correct device

    mask = torch.zeros_like(full_field, device=device)  # Ensure mask is on GPU if full_field is
    num_points = full_field.numel()  # Total number of elements in the tensor
    num_observed = int(observed_fraction * num_points)

    # Ensure indices are generated on the correct device
    indices = torch.randperm(num_points, device=device)[:num_observed]

    mask.view(-1)[indices] = 1  # Apply the mask
    partial_field = full_field * mask  # Mask the full field

    # Use two channels: one for the partial field, one for the mask
    input_tensor = torch.stack([partial_field, mask], dim=0)

    return input_tensor
