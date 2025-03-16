import torch
import random
from utils import rk4_step

def generate_simulation(nx, ny, dx, dy, t, noise_amplitude=0.05):
    """Generate a simulation with a randomized Gaussian initial condition and noise."""
    # Define grid coordinates
    x = torch.linspace(0, (nx-1)*dx, nx)
    y = torch.linspace(0, (ny-1)*dy, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Randomize Gaussian parameters:
    # Center anywhere within the domain, amplitude between 0.5 and 1.0, variance between 0.1 and 0.4.
    center_x = torch.rand(1).item() * (nx * dx)
    center_y = torch.rand(1).item() * (ny * dy)
    amplitude = 0.5 + torch.rand(1).item() * 0.5
    variance = 0.1 + torch.rand(1).item() * 0.3

    # Create the Gaussian initial condition
    initial_condition = amplitude * torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / variance)

    # Create block-wise noise:
    segment_size = 10  # Dividing the grid into segments (e.g., 100/10 = 10 segments)
    num_segments_x = nx // segment_size
    num_segments_y = ny // segment_size
    noise_segment = noise_amplitude * torch.randn(num_segments_x, num_segments_y)
    # Upsample the noise: each segment's noise is repeated over a block
    noise = noise_segment.repeat_interleave(segment_size, dim=0).repeat_interleave(segment_size, dim=1)
    
    # Combine the initial condition with the noise
    initial_with_noise = initial_condition + noise
    return initial_with_noise

def simulate_simulation(nx, ny, dx, dy, nt=100, dt=0.001, noise_amplitude=0.05):
    """
    Simulates the heat equation starting from a noisy initial condition.
    Returns a tensor T of shape (nt, nx, ny) representing the time evolution.
    """
    # Get the initial condition with noise
    T = torch.empty(nt, nx, ny)
    T[0] = generate_simulation(nx, ny, dx, dy, 0, noise_amplitude)
    
    # Define the diffusion coefficient (can be made heterogeneous if desired)
    D = torch.ones_like(T[0]) * 0.1
    
    # Time integration loop using RK4
    for t in range(nt - 1):
        T[t + 1] = rk4_step(T[t], D, dx, dy, dt)
    return T
