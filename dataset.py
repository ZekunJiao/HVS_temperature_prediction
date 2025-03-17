import torch
import random
from torch.utils.data import Dataset
from utils import create_masked_input
from simulation import simulate_simulation

class TemperatureDataset(Dataset):
    def __init__(self, num_simulations, nx, ny, dx, dy, nt=100, dt=0.00005, noise_amplitude=0.05):
        # We'll store snapshots from time steps t = 1, 2, ..., nt-1 for each simulation
        snapshots = []
        for i in range(num_simulations): 
            print(f"generating simulation {i}")
            # Run the simulation to get the full time series T
            T_series = simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude)
            # Exclude the initial condition (t=0) and add remaining snapshots
            rand_idx = random.randint(1, nt - 1)
            snapshots.append(T_series[rand_idx])
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
