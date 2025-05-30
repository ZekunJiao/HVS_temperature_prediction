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

def create_masked_input(full_field, observed_fraction):
    """Create a binary mask and the corresponding partial field, ensuring CUDA compatibility."""
    device = full_field.device  # Ensure operations are done on the correct device

    mask = torch.zeros_like(full_field, device=device)  # Ensure mask is on GPU if full_field is
    num_points = int(full_field.numel() / 2)  # Total number of elements in the tensor
    num_observed = int(observed_fraction * num_points)

    # Ensure indices are generated on the correct device
    indices = torch.randperm(num_points, device=device)[:num_observed]

    mask.view(-1)[indices] = 1  # Apply the mask
    partial_field = full_field * mask  # Mask the full field

    # Use two channels: one for the partial field, one for the mask
    input_tensor = torch.stack([partial_field, mask], dim=0)
    return input_tensor

def create_operator_input_2(full_field, observed_fraction):
    """Create a binary mask and the corresponding partial field, ensuring CUDA compatibility."""
    device = full_field.device  # Ensure operations are done on the correct device

    mask = torch.zeros_like(full_field, device=device)  # Ensure mask is on GPU if full_field is
    num_points = int(full_field.numel() / 2)  # Total number of elements in the tensor
    num_observed = int(observed_fraction * num_points)

    # Ensure indices are generated on the correct device
    indices = torch.randperm(num_points, device=device)[:num_observed]

    mask.view(-1)[indices] = 1  # Apply the mask

    x = torch.nonzero(mask).reshape(2, -1).float()
    u = full_field.view(-1)[indices]  # Mask the full field
    print(x.shape)

    return x, u


def create_x(T_input, observed_fraction, domain_fraction):
    """
    Create a partial input for an operator by randomly sampling a fraction of points
    from the top half of a 2D tensor (full_field of shape (H, W)).

    Args:
        full_field (torch.Tensor): A 2D tensor of shape (H, W).
        observed_fraction (float): Fraction of points to sample from the top half.

    Returns:
        x (torch.Tensor): Tensor of shape (2, num_observed) containing the (col, row) coordinates.
    """
    device = T_input.device
    H, W = T_input.shape

    # Restrict to the top half of the field
    H_sample = int(H * domain_fraction)
    max_index = H_sample * W
    num_observed = int(observed_fraction * max_index)

    # Generate random flat indices from 0 to num_points_restricted - 1 (min=0, max=num_point_observed)
    # Take only the observed fraction
    indices = torch.randperm(max_index, device=device)[:num_observed]

    # Convert flat indices to 2D coordinates (row, col)
    row_indices = indices // W   # row index in range [0, H_sample)
    col_indices = indices % W

    # Stack to form coordinates: first row is column indices, second row is row indices
    norm_factor = max(H, W) - 1
    col_indices_norm = col_indices.float() / (W - 1)
    row_indices_norm = row_indices.float() / (H - 1)
    # Stack normalized coordinates: first row is normalized column indices, second row is normalized row indices
    x = torch.stack([col_indices_norm, row_indices_norm])  # Shape: (2, num_observed)
    print("x shape", x.shape)
    return x



