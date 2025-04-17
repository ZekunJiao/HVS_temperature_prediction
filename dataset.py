from typing import Tuple
import torch
import random
from torch.utils.data import Dataset
from utils import create_masked_input, create_operator_input
from simulation import simulate_simulation
import os
import numpy as np
import matplotlib.pyplot as plt
from continuiti.data import OperatorDataset


class TemperatureDataset(Dataset):
    def __init__(self, num_simulations, nx, ny, dx, dy, nt, dt, noise_amplitude, save_path=None):
        # We'll store snapshots from time steps t = 1, 2, ..., nt-1 for each simulation
        self.data = []
        for i in range(num_simulations): 
            print(f"generating simulation {i}")
            # Run the simulation to get the full time series T
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            T_series = simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude, device=device)
            # Exclude the initial condition (t=0) and add remaining snapshots
            rand_timeframe = random.randint(1, nt - 1)
            self.data.append(T_series[nt - 1].cpu())
            del T_series
            torch.cuda.empty_cache()

        # Stack all snapshots into a tensor of shape (num_samples, nx, ny)
        self.data = torch.stack(self.data, dim=0)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self, save_path)
            print(f"Dataset saved at {save_path}")

        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        full_field = self.data[idx]  # shape: (nx, ny)
        _, input_tensor = create_masked_input(full_field, observed_fraction=0.02)
        # Add a channel dimension to full_field (target is one-channel)
        target = full_field.unsqueeze(0)
        return input_tensor, target


import torch
import torch.utils.data as td
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from continuiti.transforms import Transform
from continuiti.operators.shape import OperatorShapes, TensorShape

class OperatorDataset(td.Dataset):
    """A dataset for operator training.
 
    In operator training, at least one function is mapped onto a second one. To fulfill the properties discretization
    invariance, domain independence and learn operators with physics-based loss access to at least four different
    discretized spaces is necessary. One on which the input is sampled (x), the input function sampled on these points
    (u), the discretization of the output space (y), and the output of the operator (v) sampled on these points. Not
    all loss functions and/or operators need access to all of these attributes.
 
    Args:
        x: Tensor of shape (num_observations, x_dim, num_sensors...) with sensor positions.
        u: Tensor of shape (num_observations, u_dim, num_sensors...) with evaluations of the input functions at sensor positions.
        y: Tensor of shape (num_observations, y_dim, num_evaluations...) with evaluation positions.
        v: Tensor of shape (num_observations, v_dim, num_evaluations...) with ground truth operator mappings.
 
    Attributes:
        shapes: Shape of all tensors.
        transform: Transformations for each tensor.
    """
 
    def __init__(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        x_transform: Optional[Transform] = None,
        u_transform: Optional[Transform] = None,
        y_transform: Optional[Transform] = None,
        v_transform: Optional[Transform] = None,
    ):
        # get dimensions and sizes
        x_dim, x_size = x.size(1), x.size()[2:]
        u_dim, u_size = u.size(1), u.size()[2:]
        y_dim, y_size = y.size(1), y.size()[2:]
        v_dim, v_size = v.size(1), v.size()[2:]

 
        super().__init__()
 
        self.x = x
        self.u = u
        self.y = y
        self.v = v
 
        # used to initialize architectures
        self.shapes = OperatorShapes(
            x=TensorShape(dim=x_dim, size=x_size),
            u=TensorShape(dim=u_dim, size=u_size),
            y=TensorShape(dim=y_dim, size=y_size),
            v=TensorShape(dim=v_dim, size=v_size),
        )
 
        self.transform = {
            dim: tf
            for dim, tf in [
                ("x", x_transform),
                ("u", u_transform),
                ("y", y_transform),
                ("v", v_transform),
            ]
            if tf is not None
        }
 
    def __len__(self) -> int:
        """Return the number of samples.
 
        Returns:
            Number of samples in the entire set.
        """
        return self.x.size(0)
 
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves the input-output pair at the specified index and applies transformations.
 
        Parameters:
            idx: The index of the sample to retrieve.
 
        Returns:
            A tuple containing the three input tensors and the output tensor for the given index.
        """
        return self._apply_transformations(
            self.x[idx], self.u[idx], self.y[idx], self.v[idx]
        )
 
    def _apply_transformations(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies class transformations to four tensors.
 
        Args:
            x: Tensor of shape (num_observations, x_dim, num_sensors...) with sensor positions.
            u: Tensor of shape (num_observations, u_dim, num_sensors...) with evaluations of the input functions at sensor positions.
            y: Tensor of shape (num_observations, y_dim, num_evaluations...) with evaluation positions.
            v: Tensor of shape (num_observations, v_dim, num_evaluations...) with ground truth operator mappings.
 
        Returns:
            Input samples with class transformations applied.
        """
        sample = {"x": x, "u": u, "y": y, "v": v}
 
        # transform
        for dim, val in sample.items():
            if dim in self.transform:
                sample[dim] = self.transform[dim](val)
 
        return sample["x"], sample["u"], sample["y"], sample["v"]



class OperatorTemperatureDataset(OperatorDataset):
    def __init__(self, num_simulations, nx, ny, dx, dy, nt, dt, noise_amplitude,observed_fraction, save_path=None):
        # We'll store snapshots from time steps t = 1, 2, ..., nt-1 for each simulation
        x_data = []
        u_data = []
        y_data = []
        v_data = []

        for i in range(num_simulations): 
            print(f"generating simulation {i}")
            # Run the simulation to get the full time series T
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            T_series = simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude, device=device)
            # Exclude the initial condition (t=0) and add remaining snapshots
            rand_timeframe = random.randint(1, nt - 1)

            ## min max scale 

            v = T_series[nt - 1].cpu()
            v = (v - torch.min(v)) / (torch.max(v) - torch.min(v))

            fig, axs = plt.subplots(1, 3, figsize=(22,6))
            
            
            x, u = create_operator_input(v, observed_fraction=observed_fraction)
            grid_x, grid_y = torch.meshgrid(torch.arange(0, len(v), dtype=torch.float32), torch.arange(0, len(v[0]), dtype=torch.float32))
            
            # Normalize the grid coordinates to [0,1]
            norm_factor = max(nx, ny) -1
            grid_x = grid_x / (nx - 1)
            grid_y = grid_y / (ny - 1)
            y = torch.stack([grid_y, grid_x])

            # ######### plotting ##########
            # im = axs[0].imshow(v, cmap="viridis", origin="lower")
            # fig.colorbar(im, ax=axs[0], label="Temperature")
            # axs[0].set_title("Temperature Field at Final Time")
            # axs[0].set_xlabel("X index")
            # axs[0].set_ylabel("Y index")

            # axs[1].scatter(x[0], x[1], c=u, cmap="viridis", s=1)
            # axs[1].set_xlim(0, 2)
            # axs[1].set_ylim(0, 2)

            # axs[2].scatter(y[0], y[1], c=v, cmap="viridis", s=1)
            # axs[2].set_xlim(0, 2)
            # axs[2].set_ylim(0, 2)
            # plt.show()
            # ########plotting ends ##########

            x_data.append(x)
            y_data.append(y)
            u_data.append(u)
            v_data.append(v)

            del T_series
            torch.cuda.empty_cache()
        # Stack all snapshots into a tensor of shape (num_samples, nx, ny)
        
        x_data = torch.stack(x_data, dim=0) 
        u_data = torch.stack(u_data, dim=0)
        y_data = torch.stack(y_data, dim=0) 
        v_data = torch.stack(v_data, dim=0)
        v_data = v_data.unsqueeze(dim=1)
        u_data = u_data.unsqueeze(dim=1)

        print(f"x shape: {x_data.shape}")
        print(f"y shape: {y_data.shape}")
        print(f"u shape: {u_data.shape}")
        print(f"v shape: {v_data.shape}")

        super().__init__(x=x_data, y=y_data, u=u_data, v=v_data)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self, save_path)
            print(f"Dataset saved at {save_path}")
        

if __name__ == "__main__": 
    '''
    Generate dataset here
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script folder
    os.chdir(script_dir)  # Set script directory as working directory

    nx, ny = 10, 10
    dx, dy = 0.1, 0.1
    num_simulations = 5000
    observed_fraction = 0.5
    save_path = os.path.join(script_dir, "datasets", f"operator_dataset_{num_simulations}_observed{observed_fraction}_nx{nx}_ny{ny}_nomalized_full.pt")

    print(save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        print("no such path")
        exit()

    print(num_simulations)
    dataset = OperatorTemperatureDataset(num_simulations, nx, ny, dx, dy, nt=300, dt=0.0001, observed_fraction=observed_fraction, noise_amplitude=0, save_path=save_path)

    print(f"Dataset size: {len(dataset)} samples")