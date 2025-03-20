import torch
import random
from utils import rk4_step
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_simulation(nx, ny, dx, dy, noise_amplitude=0.00, device='cpu'):
    """Generate a simulation with a randomized Gaussian initial condition and noise on GPU."""
    # Define grid coordinates
    x = torch.linspace(0, (nx-1)*dx, nx, device=device)
    y = torch.linspace(0, (ny-1)*dy, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    length_x = nx * dx
    length_y = ny * dy

    # Randomized Gaussian parameters (placed on GPU)
    center_x = torch.rand(1, device=device).item() * length_x
    center_y = torch.rand(1, device=device).item() * length_y

    # center_x = torch.rand(1, device=device).item() * (0.9 - 0.1) * length_x + 0.1 * length_x
    # center_y = torch.rand(1, device=device).item() * (0.9 - 0.1) * length_y + 0.1 * length_y

    amplitude = 1 + torch.rand(1, device=device).item() * 0.5

    variance_x = torch.rand(1, device=device).item() * 3 # Variance in x-direction (controls horizontal spread)
    variance_y = torch.rand(1, device=device).item() * 3  # Variance in y-direction (controls vertical spread)

    # Compute Gaussian function with different variances
    initial_condition = amplitude * torch.exp(-(((X - center_x)**2 / variance_x) + ((Y - center_y)**2 / variance_y)))
    print("center x", center_x, "center y", center_y, "variance x", variance_x, "variance y", variance_y)

    # Create block-wise noise (on GPU)
    segment_size = 2
    num_segments_x = nx // segment_size
    num_segments_y = ny // segment_size
    noise_segment = noise_amplitude * torch.randn(num_segments_x, num_segments_y, device=device)
    
    # Upsample the noise
    noise = noise_segment.repeat_interleave(segment_size, dim=0).repeat_interleave(segment_size, dim=1)

    # Combine initial condition with noise (all on GPU)
    initial_with_noise = initial_condition + noise
    return initial_with_noise

def simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude=0.00, device='cpu'):
    """
    Simulates the heat equation using RK4, fully utilizing CUDA.
    Returns a tensor T of shape (nt, nx, ny) stored on `device`.
    """
    # Allocate tensor on GPU
    T = torch.empty(nt, nx, ny, device=device)

    # Initialize first time step
    T[0] = generate_simulation(nx, ny, dx, dy, noise_amplitude, device=device)

    # Define the diffusion coefficient (D) on GPU
    D = torch.ones_like(T[0], device=device) * 0.3
    # D[40:50, 0:99] = 0.06  # Specific region diffusion coefficient

    # Time integration loop using RK4
    for t in range(nt - 1):
        T[t + 1] = rk4_step(T[t], D, dx, dy, dt)  # Ensure `rk4_step` is CUDA-compatible

    return T  # Returns a CUDA tensor if `device='cuda'

def visualize_simulation(T, timesteps=[0, 10, 50, -1], device='cpu'):
    """
    Visualizes the simulation results at specified timesteps.
    - T: Tensor (nt, nx, ny), simulation results.
    - timesteps: List of time indices to visualize.
    - device: Device used for computation.
    """
    T = T.to('cpu')  # Move to CPU for visualization

    num_steps = len(timesteps)
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 5))

    for i, t in enumerate(timesteps):
        ax = axes[i]
        ax.set_title(f"Timestep {t if t >= 0 else len(T)-1}")
        img = ax.imshow(T[t].numpy(), cmap='viridis', origin='lower')
        plt.colorbar(img, ax=ax)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simulation parameters
    nx, ny = 100, 100  # Number of grid points
    dx, dy = 0.01, 0.01  # Grid spacing
    dt = 0.0001  # Time step
    nt = 800  # Number of time steps     # Time step size

    # Run the simulation
    T = simulate_simulation(nx, ny, dx, dy, nt=nt, dt=dt, device=device)
    vmin = torch.min(T).cpu().item()
    vmax = torch.max(T).cpu().item()

    # Visualize results at key time points
    # visualize_simulation(T, timesteps=[0, 10, 50, -1], device=device)

    import numpy as np
    from IPython.display import HTML
    import matplotlib.animation as animation

    plt.figure(figsize=(6, 5))
    plt.imshow(T[0].cpu().numpy().T, cmap='viridis', origin="lower")
    plt.colorbar(label='Temperature')
    plt.title("Initial Condition (t = 0)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.imshow(np.zeros((nx, ny)), cmap='viridis', vmin=vmin, vmax=vmax, origin="lower")  # Initialize empty frame
    cbar = plt.colorbar(cp)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Simulation at frame ')
    time_steps = torch.linspace(0, nt-1, 100)  

    # Function to update the frame
    T = T.cpu()
    def update(frame):
        # t = time_steps[frame] * torch.ones(x_eval.shape)  # Update time
        # pred = model(torch.vstack([x_eval, y_eval, t]).T)  # Get predictions
        # pred = pred.reshape(40, 24).T.detach().numpy()  # Reshape for plotting
        cp.set_data(T[time_steps[frame].long(), :, :].T)
        ax.set_title(f'Simulation at frame {frame}')  # Update title
        for c in ax.collections:
            c.remove()
        return [cp]

    ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=30, blit=False)

    # Display animation in Jupyter Notebook
    plt.show()
