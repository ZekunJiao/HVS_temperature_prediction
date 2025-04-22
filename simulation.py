import torch
import random
from utils import rk4_step
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def initialize_simulation(nx, ny, dx, dy, noise_amplitude=0.0, device='cpu'):
    """
    Generate a randomized 2D Gaussian initial condition (with optional block noise).

    Args:
        nx, ny (int): number of grid points in width (columns, w) and height (rows, h).
        dx, dy (float): grid spacing in w and h.
        noise_amplitude (float): amplitude of added block noise.
        device (str): torch device specifier.

    Returns:
        torch.Tensor of shape (ny, nx) on `device`.
    """
    # 1D coordinates
    w = torch.linspace(0, (nx - 1) * dx, nx, device=device)   # width axis (columns)
    h = torch.linspace(0, (ny - 1) * dy, ny, device=device)   # height axis (rows)

    # 2D mesh (h_grid indexed by rows, w_grid by columns)
    h_grid, w_grid = torch.meshgrid(h, w, indexing='ij')  # both shape (ny, nx)

    # random Gaussian parameters
    center_w = random.random() * w_grid.max().item()
    center_h = random.random() * h_grid.max().item()
    amplitude = 1.0 + random.random() * 0.5
    variance_w = random.random() * 3.0
    variance_h = random.random() * 3.0

    # Gaussian bump
    initial = amplitude * torch.exp(
        -(((w_grid - center_w) ** 2) / variance_w
          + ((h_grid - center_h) ** 2) / variance_h)
    )

    # block‐wise noise
    if noise_amplitude > 0.0:
        seg = 2
        nw = nx // seg
        nh = ny // seg
        noise_block = noise_amplitude * torch.randn(nh, nw, device=device)
        noise = noise_block.repeat_interleave(seg, dim=0).repeat_interleave(seg, dim=1)
        initial = initial + noise

    return initial


def simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude=0.0, device='cpu') -> torch.Tensor:
    """
    Simulates the heat equation via RK4.

    Returns:
        T: Tensor of shape (nt, ny, nx) on `device`.
    """
    T = torch.empty(nt, ny, nx, device=device)
    T[0] = initialize_simulation(nx, ny, dx, dy, noise_amplitude, device=device)

    # diffusion coefficient (same shape as one slice)
    D = torch.full((ny, nx), 0.3, device=device)

    for t in range(nt - 1):
        T[t + 1] = rk4_step(T[t], D, dx, dy, dt)

    return T


def visualize_simulation(T, timesteps=[0, 10, 50, -1]):
    """
    Visualizes T (nt, ny, nx) at the given timesteps.
    """
    T_cpu = T.cpu()
    fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 5))

    for ax, step in zip(axes, timesteps):
        idx = step if step >= 0 else T_cpu.shape[0] - 1
        im = ax.imshow(T_cpu[idx].numpy(), cmap='viridis', origin='lower')
        ax.set_title(f"Step {idx}")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # grid + time parameters
    nx, ny = 100, 200       # columns, rows
    dx, dy = 0.01, 0.01
    nt, dt = 800, 1e-4

    # run
    T = simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude=0.0, device=device)

    # show initial condition
    plt.figure(figsize=(6,5))
    plt.imshow(T[0].cpu().numpy(), cmap='viridis', origin='lower')
    plt.colorbar(label='Temperature')
    plt.title("Initial Condition (t=0)")
    plt.xlabel("w (columns)")
    plt.ylabel("h (rows)")
    plt.show()

    # animation
    import numpy as np
    from matplotlib import animation

    fig, ax = plt.subplots(figsize=(8,6))
    vmin, vmax = T.min().item(), T.max().item()
    img = ax.imshow(T[0].cpu().numpy(), cmap='viridis',
                    origin='lower', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img, ax=ax)
    ax.set_xlabel("w (columns)")
    ax.set_ylabel("h (rows)")

    def update(frame):
        data = T[frame].cpu().numpy()
        img.set_data(data)
        ax.set_title(f"Simulation at step {frame}")
        return [img]

    frames = np.linspace(0, nt-1, 100, dtype=int)
    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=30, blit=True
    )
    plt.show()
