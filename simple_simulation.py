import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import random

def gradient(T, dx, dy):
    """Gradient part of the PDE."""
    T_x = (torch.roll(T, shifts=-1, dims=0) - torch.roll(T, shifts=1, dims=0)) / (2 * dx)
    T_y = (torch.roll(T, shifts=-1, dims=1) - torch.roll(T, shifts=1, dims=1)) / (2 * dy)
    return T_x, T_y

def laplacian(T, dx, dy):
    """Laplacian part of the pde."""
    T_xx = (torch.roll(T, shifts=-1, dims=0) - 2 * T + torch.roll(T, shifts=1, dims=0)) / dx**2
    T_yy = (torch.roll(T, shifts=-1, dims=1) - 2 * T + torch.roll(T, shifts=1, dims=1)) / dy**2

    return T_xx + T_yy

def heat_equation_rhs(T, D, dx, dy):
    """Heat equation RHS."""
    T_x, T_y = gradient(T, dx, dy)
    D_x, D_y = gradient(D, dx, dy)
    
    diffusion_term = D * laplacian(T, dx, dy)
    gradient_term = D_x * T_x + D_y * T_y
    
    return diffusion_term + gradient_term

def rk4_step(T, D, dx, dy, dt):
    """RK4 time stepping scheme."""
    k1 = heat_equation_rhs(T, D, dx, dy)
    k2 = heat_equation_rhs(T + 0.5 * dt * k1, D, dx, dy)
    k3 = heat_equation_rhs(T + 0.5 * dt * k2, D, dx, dy)
    k4 = heat_equation_rhs(T + dt * k3, D, dx, dy)
    return T + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# Function to create a binary mask and the corresponding partial field
def create_masked_input(full_field, observed_fraction=0.05):
    mask = torch.zeros_like(full_field)
    num_points = full_field.numel()
    num_observed = int(observed_fraction * num_points)
    indices = torch.randperm(num_points)[:num_observed]
    mask.view(-1)[indices] = 1
    partial_field = full_field * mask
    # Use two channels: one for the partial field and one for the mask
    input_tensor = torch.stack([partial_field, mask], dim=0)
    return input_tensor

# Function to generate a simulation with a randomized Gaussian initial condition and noise
def generate_simulation(nx, ny, dx, dy, t, noise_amplitude=0.05):
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
    T[0] = generate_simulation(nx, ny, dx, dy, noise_amplitude)
    
    # Define the diffusion coefficient (can be made heterogeneous if desired)
    D = torch.ones_like(T[0]) * 0.1
    
    # Time integration loop using RK4
    for t in range(nt - 1):
        T[t + 1] = rk4_step(T[t], D, dx, dy, dt)
    return T

# Create a dataset that generates a diverse set of simulations
class TemperatureDataset(Dataset):
    def __init__(self, num_simulations, nx, ny, dx, dy, nt=100, dt=0.001, noise_amplitude=0.05):
        # We'll store snapshots from time steps t = 1, 2, ..., nt-1 for each simulation
        snapshots = []
        for i in range(10): 
            print(f"generating simulation {i}")
            # Run the simulation to get the full time series T
            T_series = simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude)
            # Exclude the initial condition (t=0) and add remaining snapshots
            print(T_series[random.randint(1, nt - 1)])
            snapshots.append(T_series[random.randint(1, nt - 1)])
        # Stack all snapshots into a tensor of shape (num_samples, nx, ny)
        self.data = torch.stack(snapshots, dim=0)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        full_field = self.data[idx]  # shape: (nx, ny)
        input_tensor = create_masked_input(full_field, observed_fraction=0.05)
        # Add a channel dimension to full_field (target is one-channel)
        target = full_field.unsqueeze(0)
        return input_tensor, target

# Define a simple CNN model (Proof-of-Concept)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# Set grid and dataset parameters
nx, ny = 100, 100
dx, dy = 0.01, 0.01
num_simulations = 1000  # Number of different simulation samples

# Create dataset and dataloader
dataset = TemperatureDataset(num_simulations, nx, ny, dx, dy)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Create the full dataset
# Define the split ratio (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Randomly split the dataset into train and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders for the training and testing sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set up model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50  # You can adjust the number of epochs
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

# (Optional) Visualize one example from the dataset and its prediction
model.eval()
for i in range(len(test_dataset)):
    if i > 5:
        break
    input_sample, target_sample = test_dataset[i]
    with torch.no_grad():
        prediction = model(input_sample.unsqueeze(0).to(device)).cpu().squeeze()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Partial Input")
    plt.imshow(input_sample[0], cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(target_sample.squeeze(), cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(prediction, cmap='viridis')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
