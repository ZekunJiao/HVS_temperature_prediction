from typing import Tuple
import torch
import random
from torch.utils.data import Dataset
from utils import create_masked_input, create_x
from simulation import create_blob_diffusivity, simulate_simulation
import os
import numpy as np
import matplotlib.pyplot as plt
from continuiti.data import OperatorDataset
import datetime


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
    def __init__(self, num_simulations, nx, ny, dx, dy, nt, dt, t0, noise_amplitude,observed_fraction, device, save_path=None):
        # We'll store snapshots from time steps t = 1, 2, ..., nt-1 for each simulation
        x_data = []
        u_data = []
        y_data = []
        v_data = []

        for i in range(num_simulations): 
            print(f"generating simulation {i}")
            # Run the simulation to get the full time series T

            T_series = simulate_simulation(nx, ny, dx, dy, nt, dt, noise_amplitude, device=device)
            # Exclude the initial condition (t=0) and add remaining snapshots
            rand_timeframe = random.randint(1, nt - 1)

            ## min max scale 

            v = T_series[nt - 1]
            v = (v - torch.min(v)) / (torch.max(v) - torch.min(v))

            x, u = create_operator_input(v, observed_fraction=observed_fraction)
            grid_x, grid_y = torch.meshgrid(torch.arange(0, len(v), dtype=torch.float32), torch.arange(0, len(v[0]), dtype=torch.float32))
            
            # Normalize the grid coordinates to [0,1]
            norm_factor = max(nx, ny) -1
            grid_x = grid_x / (nx - 1)
            grid_y = grid_y / (ny - 1)
            y = torch.stack([grid_y, grid_x])

            ######### plotting ##########
            # fig, axs = plt.subplots(1, 3, figsize=(22,6))
            # im = axs[0].imshow(v, cmap="viridis", origin="lower")
            # fig.colorbar(im, ax=axs[0], label="Temperature")
            # axs[0].set_title("Temperature Field at Final Time")
            # axs[0].set_xlabel("X index")
            # axs[0].set_ylabel("Y index")

            # axs[1].scatter(x[0], x[1], c=u, cmap="viridis", s=1)
            # axs[1].set_xlim(0, 2)
            # axs[1].set_ylim(0, 2)
            print(y.shape)
            print(v.shape)

            # axs[2].scatter(y[0], y[1], c=v, cmap="viridis", s=1)
            # axs[2].set_xlim(0, 2)
            # axs[2].set_ylim(0, 2)
            # plt.savefig("./temp")
            ########plotting ends ##########

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


class SimulationDataset(td.Dataset):
    def __init__(      
        self, 
        num_simulations, 
        nx, ny, dx, dy, nt, dt, 
        d_in, d_out,
        start_x, start_y, end_x, end_y,
        t0, 
        noise_amplitude, 
        device,
        save_path
    ):
        
        super().__init__()
        inputs = []
        outputs = []
        for i in range(num_simulations):
            print(f"generating simulation {i}")
            T_series = simulate_simulation(nx, ny, dx, dy, nt, dt,
                                           d_in=d_in, d_out=d_out,
                                             start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y,
                                            noise_amplitude=noise_amplitude, device=device)
            inputs.append(T_series[t0].cpu())
            outputs.append(T_series[nt-1].cpu())

            # plt.figure()  # Create a new figure for the initial temperature field
            # plt.imshow(T_series[t0].cpu(), cmap="viridis", origin="lower")
            # plt.colorbar(label="Temperature")
            # plt.title(f"Temperature Field at Time {t0*dt:.3f}s")
            # plt.savefig(f"./temp_init.png")

            # plt.figure()  # Create a new figure for the final temperature field
            # plt.imshow(T_series[nt-1].cpu(), cmap="viridis", origin="lower")
            # plt.colorbar(label="Temperature")  
            # plt.title(f"Temperature Field at Time {nt*dt:.3f}s")
            # plt.savefig(f"./temp_final.png")
            # plt.close()  # Close the figure to free memory

            del T_series
            torch.cuda.empty_cache()
        self.inputs = torch.stack(inputs)
        self.outputs = torch.stack(outputs)

        torch.save({
            'inputs':  self.inputs,
            'outputs': self.outputs
        }, save_path)

    def __getitem__(self, index): 
        return (self.inputs[index], self.outputs[index])

    def __len__(self):
        return self.inputs.shape[0]


class FullSimulationDataset(td.Dataset):
    def __init__(      
        self, 
        num_simulations, 
        nx, ny, dx, dy, nt, dt, D,
        noise_amplitude, 
        device,
        save_path
    ):
        
        super().__init__()
        data = []
        for i in range(num_simulations):
            print(f"generating simulation {i}")
            T_series = simulate_simulation(nx, ny, dx, dy, nt, dt,D=D,
                                            noise_amplitude=noise_amplitude, device=device)

            # Display T_series at nt/2
            mid_step = nt // 2
            plt.figure()
            plt.imshow(T_series[mid_step].cpu().numpy(), cmap='viridis', origin='lower')
            plt.colorbar(label="Temperature")
            plt.title(f"Temperature Field at time step {mid_step} (nt/2)")
            plt.xlabel("X index")
            plt.ylabel("Y index")
            plt.savefig(f"simulation_frame_at_nt_half_sim_{i}.png") # Save the figure
            plt.close() # Close the figure to free memory
            
            data.append(T_series.cpu())
            del T_series
            torch.cuda.empty_cache()
        self.data = torch.stack(data)
        torch.save(self.data, save_path)

    def __getitem__(self, index): 
        return (self.data[index], self.data[index])

    def __len__(self):
        return self.data.shape[0]


class OperatorFieldMappingDataset(OperatorDataset):
    def __init__(
        self, 
        num_samples,
        sensor_coordinates, 
        observed_fraction, 
        domain_fraction, 
        simulation_file_path,
        save_path
    ):
        '''
        Args:
            num_samples: Number of samples to generate.
            sensor_coordinates: Coordinates of the sensors. (2, num_sensors) [row, col]
            observed_fraction: Fraction of the field to be observed.
            domain_fraction: Fraction of the domain to be observed.
            simulation_file_path: Path to the simulation dataset file.
            save_path: Path to save the generated dataset.
        '''
        
        simulation_dataset = torch.load(simulation_file_path, map_location="cpu")

        inputs = simulation_dataset["inputs"]
        outputs = simulation_dataset["outputs"]
        N, H, W = inputs.shape

        if num_samples > N:
            raise ValueError(f"num_samples ({num_samples}) cannot exceed available samples ({N}).")
            
        print("input shape", inputs[0].shape)
        print(f" ############## Loading simulation dataset: {simulation_file_path}, size: {N} ##################")

        u_data = []
        v_data = []

        h_grid, w_grid = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        # normalize the coordinates
        y = torch.stack([h_grid, w_grid]) # shape: (2, H, W)

        if sensor_coordinates is not None:
            # Use the provided sensor coordinates
            x = sensor_coordinates
        else:
            x = create_x(T_input=inputs[0], observed_fraction=observed_fraction, domain_fraction=domain_fraction)

        for i in range(num_samples):
            cols = (x[0] * (W - 1)).round().long()  # x[0] → columns, scale by W-1
            rows = (x[1] * (H - 1)).round().long()  # x[1] → rows,    scale by H-1

            u = inputs[i, rows, cols]  
            
            v = outputs[i]
            # normalize function values
            u_max = torch.max(u)
            u_min = torch.min(u)
            v_max = torch.max(v)
            v_min = torch.min(v)
            u = (u - u_min) / (u_max- u_min)
            v = (v - v_min) / (v_max - v_min)

            ####### plotting ##########
            # fig, ax = plt.subplots(1, 4, figsize=(20, 6))

            # scatter_1 = ax[0].scatter(
            #     x[0].cpu(), 
            #     x[1].cpu(),
            #     c=u.cpu(), 
            #     cmap="viridis", 
            #     vmin=0,
            #     vmax=1,
            # )
            # ax[0].set_aspect("equal")
            # cbar1 = fig.colorbar(scatter_1, ax=ax[0])
            # cbar1.set_label("u value")

            # scatter_2 = ax[1].scatter(
            #     y[1].cpu(), 
            #     y[0].cpu(),
            #     c=v.cpu(), 
            #     vmin=0,
            #     vmax=1,
            #     cmap="viridis", 
            # )
            # ax[1].set_aspect("equal")
            # cbar2 = fig.colorbar(scatter_2, ax=ax[1])
            # cbar2.set_label("v value")

            # im_1 = ax[2].imshow(
            #     inputs[i],
            #     cmap="viridis", 
            #     origin="lower",
            #     vmin=0,
            #     vmax=1,
            # )
            # ax[2].set_aspect("equal")
            # cbar4 = fig.colorbar(im_1, ax=ax[2])
            # cbar4.set_label("v value")

            # im_2 = ax[3].imshow(
            #     outputs[i],
            #     cmap="viridis", 
            #     origin="lower",
            #     vmin=0,
            #     vmax=1,
            # )
            # ax[3].set_aspect("equal")
            # cbar4 = fig.colorbar(im_2, ax=ax[3])
            # cbar4.set_label("v value")

            # plt.tight_layout()
            # plt.savefig(f"./temp.png")
            # plt.show()
            ###### end plotting ########
            u_data.append(u)
            v_data.append(v)    

        x = x.unsqueeze(0).expand(num_samples, *x.shape)
        y = y.unsqueeze(0).expand(num_samples, *y.shape)
        u = torch.stack(u_data)
        v = torch.stack(v_data)

        u = torch.unsqueeze(u, dim=1)
        v = torch.unsqueeze(v, dim=1)

        super().__init__(u=u, v=v, x=x, y=y)

        if save_path:
            torch.save(self, save_path)
            print(f"Dataset saved at {save_path}")


if __name__ == "__main__": 
    '''
    Generate dataset here
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script folder
    os.chdir(script_dir)  # Set script directory as working directory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    nx, ny = 100, 100
    dx, dy = 0.01, 0.01
    num_simulations = 10
    nt = 300
    t0 = nt - 1
    d_min = 0.1
    d_max = 0.3
    start_y = random.randint(0, int(nx / 2) - 1)
    end_y = random.randint(start_y, nx - 1)    
    start_x = random.randint(0, int(ny / 2) - 1)
    end_x = random.randint(start_x, ny - 1)
    dt = 5e-5
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    noise_amplitude = 0.0

    save_path_simulation = os.path.join(script_dir, "datasets", "simulation", 
                                        f"{timestamp}_simulation_n{num_simulations}_t0{t0*dt:.3f}_t{nt*dt:.3f}_nx{nx}_ny{ny}"
                                        f"_dt{dt}_dmin{d_min}_dmax{d_max}_sy{start_y}_ey{end_y}_sx{start_x}_ex{end_x}.pt")
    
    print(save_path_simulation)
    if not os.path.exists(os.path.dirname(save_path_simulation)):
        print("no such path")
        exit()
    
    D = torch.full((ny, nx), d_max, device=device)
    D[start_y:end_y, start_x:end_x] = d_min
    
    D = create_blob_diffusivity(ny=ny, nx=nx, d_min=d_min, d_max=d_max, n_blobs=5, radius=20, device=device)

    plt.figure(figsize=(6, 5))
    plt.imshow(D.cpu().numpy(), cmap='viridis', origin='lower')
    plt.colorbar(label='Diffusion Coefficient')
    plt.savefig("diffusion_coefficient.png")
    plt.show()

    simulation_dataset = FullSimulationDataset(
        num_simulations=num_simulations,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        D=D,
        nt=nt,
        dt=dt,
        noise_amplitude=noise_amplitude,
        device=device,
        save_path=save_path_simulation
    )
    # ######################################################

    # ########### generate operator dataset ################

    # num_samples = 1000
    # observed_fraction = 0.1
    # domain_fraction = 0.5
    # simulation_file = "simulation_n5000_to0_t0.030_nx10_ny20.pt"
    # simulation_file_path = os.path.join(script_dir, "datasets", "simulation", simulation_file)
    # simulation_file = simulation_file.replace(".pt", "")
    # save_path = os.path.join(script_dir, "datasets", f"operator_m{num_samples}_oberserved{observed_fraction}_domain{domain_fraction}_{simulation_file}.pt")

    # dataset = OperatorFieldMappingDataset(
    #     num_samples=num_samples,
    #     observed_fraction=observed_fraction, 
    #     domain_fraction=domain_fraction,
    #     simulation_file_path=simulation_file_path,
    #     save_path=save_path
    # )

    # print(f"Dataset size: {len(dataset)} samples")
    # ######################################################