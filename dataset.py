import torch
import random
from torch.utils.data import Dataset
from utils import create_masked_input
from simulation import simulate_simulation
import os

class TemperatureDataset(Dataset):
    def __init__(self, num_simulations, nx, ny, dx, dy, nt, dt, noise_amplitude, save_path=None):
        # We'll store snapshots from time steps t = 1, 2, ..., nt-1 for each simulation
        snapshots = []
        for i in range(num_simulations): 
            print(f"generating simulation {i}")
            # Run the simulation to get the full time series T
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            T_series = simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude, device=device)
            # Exclude the initial condition (t=0) and add remaining snapshots
            rand_timeframe = random.randint(1, nt - 1)
            snapshots.append(T_series[nt - 1])
        # Stack all snapshots into a tensor of shape (num_samples, nx, ny)
        self.data = torch.stack(snapshots, dim=0)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self, save_path)
            print(f"Dataset saved at {save_path}")

        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        full_field = self.data[idx]  # shape: (nx, ny)
        input_tensor = create_masked_input(full_field, observed_fraction=0.02)
        # Add a channel dimension to full_field (target is one-channel)
        target = full_field.unsqueeze(0)
        return input_tensor, target


if __name__ == "__main__": 
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script folder
    os.chdir(script_dir)  # Set script directory as working directory

    save_path = os.path.join(script_dir, "datasets", "temperature_data.pt")
    nx, ny = 100, 100
    dx, dy = 0.01, 0.01
    num_simulations = 1000
    print(save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        print("no such path")
        exit()

    dataset = TemperatureDataset(num_simulations, nx, ny, dx, dy, nt=300, dt=0.0001, noise_amplitude=0, save_path=save_path)

    print(f"Dataset size: {len(dataset)} samples")
