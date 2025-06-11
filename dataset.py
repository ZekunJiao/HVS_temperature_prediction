import shutil
from typing import List, Tuple
import torch
import random
from torch.utils.data import Dataset
from embedding import SensorEmbedding, Voronoi
from sensors import SensorGenerator
from utils import create_masked_input, create_x
from simulation import create_blob_diffusivity, simulate_simulation
import os
import numpy as np
import matplotlib.pyplot as plt
from continuiti.data import OperatorDataset
import datetime
import torch.nn.functional as F


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


class SnapshotsSimulationDataset(td.Dataset):
    def __init__(self,      
        num_simulations, 
        random_t: bool,  # If True, generate random snapshots; if False, use the last snapshot
        nx, ny, dx, dy, nt, dt, # nt is the max number of steps for random choice
        D,  # full (ny, nx) diffusivity map to use for all sims
        # d_in, d_out, # Removed
        # start_x, start_y, end_x, end_y, # Removed
        noise_amplitude, 
        device,
        n_sensor_timestamps,
        save_path
    ):
        
        super().__init__()
        inputs = []
        outputs = []
        self.D_map_shape = D.shape # Store shape for potential reference, though D itself isn't saved in the .pt file by this class
        for i in range(num_simulations):

            # Generate a random snapshot time index (0 to nt-1)
            if random_t:
                # Weights: higher for earlier timestamps. [nt, nt-1, ..., 1] for indices [0, 1, ..., nt-1]
                time_indices = list(range(n_sensor_timestamps, nt))
                weights = [nt - i for i in time_indices] # Ensures earlier indices have higher weights
                t_snapshot = random.choices(time_indices, weights=weights, k=1)[0]
                
                # Number of time steps to simulate to get the snapshot at t_snapshot
                nt_for_simulation = t_snapshot + 1
            else:
                # Use the last snapshot (t=nt-1)
                t_snapshot = nt - 1
                nt_for_simulation = nt

            print(f"generating simulation {i} at time {t_snapshot}")
            T_series = simulate_simulation(nx, ny, dx, dy, nt_for_simulation, dt, 
                                           D=D, # Use the provided D map
                                           noise_amplitude=noise_amplitude, device=device)
            
            # T_series has shape (nt_for_simulation, ny, nx)
            
            # Collect the five frames immediately before t_snapshot:
            # indices [t_snapshot-5, t_snapshot-4, ..., t_snapshot-1]
            seq_start = t_snapshot - n_sensor_timestamps
            seq_end = t_snapshot  # slice is exclusive at the end
            input_fields = T_series[seq_start:seq_end].cpu()  # shape (5, ny, nx)
            print(f"  Input fields shape: {input_fields.shape}")
            output_field = T_series[t_snapshot].cpu()   

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

            inputs.append(input_fields)
            outputs.append(output_field)
            del T_series
            torch.cuda.empty_cache()
        self.inputs = torch.stack(inputs)
        self.outputs = torch.stack(outputs)

        torch.save({
            'inputs':  self.inputs,
            'outputs': self.outputs
            # Note: D is used for generation but not saved in this specific dataset's file.
            # If D needs to be saved, the saving logic here would need adjustment.
        }, save_path)

    def __getitem__(self, index): 
        return (self.inputs[index], self.outputs[index])

    def __len__(self):
        return self.inputs.shape[0]


class FullSimulationDataset(Dataset):
    def __init__(
        self,
        num_simulations,
        nx, ny, dx, dy, nt, dt,
        D,  # full (ny, nx) diffusivity map to use for all sims
        noise_amplitude,
        device,
        save_path=None
    ):
        super().__init__()
        self.device = device

        all_T_list = [] # Initialize an empty list
        for i in range(num_simulations):
            print(f"  Generating full simulation {i+1}/{num_simulations}")
            simulation_data = simulate_simulation(
                nx, ny, dx, dy, nt, dt,
                noise_amplitude=noise_amplitude,
                device=device,
                D=D                       # your custom full-map
            )
            all_T_list.append(simulation_data.cpu()) # Move to CPU and append

        self.T = torch.stack(all_T_list) # Stack all simulations
        self.D = D.clone().cpu()

        if save_path is not None:
            torch.save({"T": self.T, "D": self.D}, save_path)

    def __len__(self):
        # Number of simulated runs in the dataset
        return self.T.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
          T_i: Tensor of shape (nt, ny, nx)
          D  : Tensor of shape (ny, nx)
        """
        return self.T[idx], self.D

class FLRONetDataset(Dataset):
    """
    Dataset for FLRONet operator learning: sparse sensor history -> full-field snapshot.
    Each sample returns:
      sensor_tfs: (S,) int64 tensor of sensor time indices
      sensor_tensor: (S, C, H_out, W_out) embedded dense sensor fields
      full_tfs: (F,) int64 tensor of full-state time indices
      full_tensor: (F, C, H_out, W_out) ground-truth fields
    """
    def __init__(
        self,
        simulation_file: str,
        num_simulations: int,
        sensor_generator: SensorGenerator,
        embedding_generator: SensorEmbedding,
        init_sensor_timeframes: list[int],
        init_fullstate_timeframes: list[int],
        resolution: tuple[int, int],
        seed: int = 0
    ):
        # Store key parameters
        self.num_simulations = num_simulations
        self.init_sensor_timeframes = torch.tensor(init_sensor_timeframes, dtype=torch.long)
        self.init_fullstate_timeframes = torch.tensor(init_fullstate_timeframes, dtype=torch.long)
        self.S = len(self.init_sensor_timeframes)
        self.F = len(self.init_fullstate_timeframes)
        self.H_out, self.W_out = resolution

        # Generate fixed sensor positions and embedding setup
        sensor_generator.resolution = resolution
        sensor_generator.seed = seed
        self.sensor_positions = sensor_generator()
        embedding_generator.resolution = resolution
        embedding_generator.sensor_positions = self.sensor_positions
        self.embedding = embedding_generator
        if isinstance(self.embedding, Voronoi):
            _ = self.embedding.precomputed_distances

        # Setup cache directories
        base = os.path.splitext(os.path.basename(simulation_file))[0]
        self.dest = os.path.join('.', 'tensors', base)
        self.sensor_tf_dir = os.path.join(self.dest, 'sensor_timeframes')
        self.sensor_val_dir = os.path.join(self.dest, 'sensor_values')
        self.full_tf_dir = os.path.join(self.dest, 'fullstate_timeframes')
        self.full_val_dir = os.path.join(self.dest, 'fullstate_values')
        os.makedirs(self.sensor_tf_dir, exist_ok=True)
        os.makedirs(self.sensor_val_dir, exist_ok=True)
        os.makedirs(self.full_tf_dir, exist_ok=True)
        os.makedirs(self.full_val_dir, exist_ok=True)

        # Check cache
        if not os.listdir(self.sensor_tf_dir):
            # Populate cache
            data = torch.load(simulation_file, map_location='cpu')
            T = data['T'][:self.num_simulations]

            # Build time windows
            max_sensor_t = int(self.init_sensor_timeframes.max())
            self.n_chunks = T.shape[1] - max_sensor_t
            sensor_offsets = self.init_sensor_timeframes.unsqueeze(0) + torch.arange(self.n_chunks).unsqueeze(1)
            self.sensor_timeframes = sensor_offsets.long()
            full_offsets = self.init_fullstate_timeframes.unsqueeze(0) + torch.arange(self.n_chunks).unsqueeze(1)
            self.full_timeframes = full_offsets.long()

            # Write cache
            self._write_to_disk(T)
            del T
        else:
            # Load dimensions from cache
            sensor_files = os.listdir(self.sensor_tf_dir)
            total = len(sensor_files)
            self.n_chunks = total // self.num_simulations

        print(f"Initialized: {self.num_simulations} sims x {self.n_chunks} chunks each.")

        # Visualize the first sample upon initialization
        if len(self) > 0:
            print("Visualizing the first sample of the initialized FLRONetDataset...")
            try:
                # Get the first sample
                sensor_tfs_sample, sensor_tensor_sample, full_tfs_sample, full_tensor_sample = self.__getitem__(0)

                fig, axes = plt.subplots(1, 2, figsize=(13, 5))

                # Plot sensor tensor
                ax_sensor = axes[0]
                if sensor_tensor_sample.ndim == 4 and sensor_tensor_sample.shape[0] > 0 and sensor_tensor_sample.shape[1] > 0: # Check S > 0 and C > 0
                    img_sensor = sensor_tensor_sample[0, 0, :, :].numpy() # First sensor timeframe, first channel
                    sensor_time_idx_for_title = sensor_tfs_sample[0].item() if sensor_tfs_sample.numel() > 0 and sensor_tfs_sample.ndim > 0 else "N/A"
                    im_sensor = ax_sensor.imshow(img_sensor, cmap='viridis', origin='lower', aspect='auto')
                    ax_sensor.set_title(f"First Sample: Sensor Tensor (T={sensor_time_idx_for_title})")
                    fig.colorbar(im_sensor, ax=ax_sensor, label="Value")
                else:
                    ax_sensor.set_title("First Sample: Sensor Tensor (No data or invalid shape)")
                    ax_sensor.text(0.5, 0.5, "No sensor data to display", ha='center', va='center')
                ax_sensor.set_xlabel("Width")
                ax_sensor.set_ylabel("Height")

                # Plot full tensor
                ax_full = axes[1]
                if full_tensor_sample.ndim == 4 and full_tensor_sample.shape[0] > 0 and full_tensor_sample.shape[1] > 0: # Check F > 0 and C > 0
                    img_full = full_tensor_sample[0, 0, :, :].numpy()   # First full-state timeframe, first channel
                    full_time_idx_for_title = full_tfs_sample[0].item() if full_tfs_sample.numel() > 0 and full_tfs_sample.ndim > 0 else "N/A"
                    im_full = ax_full.imshow(img_full, cmap='viridis', origin='lower', aspect='auto')
                    ax_full.set_title(f"First Sample: Full Tensor (T={full_time_idx_for_title})")
                    fig.colorbar(im_full, ax=ax_full, label="Value")
                else:
                    ax_full.set_title("First Sample: Full Tensor (No data or invalid shape)")
                    ax_full.text(0.5, 0.5, "No full-field data to display", ha='center', va='center')
                ax_full.set_xlabel("Width")
                ax_full.set_ylabel("Height")

                plt.tight_layout()
                
                fig_save_dir = "."
                os.makedirs(fig_save_dir, exist_ok=True) # Ensure the directory exists
                save_path_fig = os.path.join(fig_save_dir, "initial_sample_visualization.png")
                
                plt.savefig(save_path_fig)
                print(f"Saved initial sample visualization to {save_path_fig}")
                plt.close(fig)
            except Exception as e:
                print(f"Could not visualize initial sample: {e}")
        else:
            print("Dataset is empty, skipping visualization of the first sample.")

    def __len__(self):
        return self.num_simulations * self.n_chunks

    def __getitem__(self, idx: int):
        sim_idx = idx // self.n_chunks
        suf = str(idx).zfill(6)
        sensor_tfs = torch.load(os.path.join(self.sensor_tf_dir, f'st_{suf}.pt'))
        sensor_tensor = torch.load(os.path.join(self.sensor_val_dir, f'sv_{suf}.pt'))
        full_tfs = torch.load(os.path.join(self.full_tf_dir, f'ft_{suf}.pt'))
        full_tensor = torch.load(os.path.join(self.full_val_dir, f'fv_{suf}.pt'))
        return sensor_tfs, sensor_tensor, full_tfs, full_tensor

    def _write_to_disk(self, T):
        # Clear existing cache
        for d in [self.sensor_tf_dir, self.sensor_val_dir, self.full_tf_dir, self.full_val_dir]:
            shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        # Save each chunk
        for idx in range(self.num_simulations * self.n_chunks):
            sim_idx = idx // self.n_chunks
            chunk_idx = idx % self.n_chunks
            st = self.sensor_timeframes[chunk_idx]
            ft = self.full_timeframes[chunk_idx]

            # Sensor history
            raw_s = T[sim_idx, st].unsqueeze(1)  # T is on CPU
            resized = F.interpolate(raw_s, size=(self.H_out, self.W_out), mode='bicubic', align_corners=False) # resized is on CPU
            
            # Move resized tensor to the device of sensor_positions before passing to embedding
            embedding_input_device = self.sensor_positions.device
            resized_for_embedding = resized.unsqueeze(0).to(embedding_input_device)
            
            emb_on_device = self.embedding(resized_for_embedding, seed=idx).squeeze(0)
            emb = emb_on_device.cpu() # Move embedding result back to CPU for saving

            # Full field
            raw_f = T[sim_idx, ft].unsqueeze(1)
            full = F.interpolate(raw_f, size=(self.H_out, self.W_out), mode='bicubic', align_corners=False)

            # Write
            sfx = str(idx).zfill(6)
            torch.save(st, os.path.join(self.sensor_tf_dir, f'st_{sfx}.pt'))
            torch.save(emb, os.path.join(self.sensor_val_dir, f'sv_{sfx}.pt'))
            torch.save(ft, os.path.join(self.full_tf_dir, f'ft_{sfx}.pt'))
            torch.save(full, os.path.join(self.full_val_dir, f'fv_{sfx}.pt'))

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

        # Load the simulation dataset; now inputs has shape (N, 5, H, W)
        simulation_dataset = torch.load(simulation_file_path, map_location="cpu")
        inputs = simulation_dataset["inputs"]   # shape: (N, seq_len, H, W)
        outputs = simulation_dataset["outputs"] # shape: (N, H, W)
        N, seq_len, H, W = inputs.shape

        if num_samples > N:
            num_samples = N

        print(f" ############## Loading simulation dataset: {simulation_file_path}, size: {N} ##################")

        u_data = []
        v_data = []

        # Build the spatial grid y of shape (2, H, W)
        h_grid, w_grid = torch.meshgrid(
            torch.linspace(0, 1, H), 
            torch.linspace(0, 1, W), 
            indexing="ij"
        )
        y = torch.stack([h_grid, w_grid])  # shape: (2, H, W)

        # Determine sensor locations x of shape (2, num_sensors)
        if sensor_coordinates is not None:
            x = sensor_coordinates  # assume already normalized in [0,1]
        else:
            # Use the first time‐slice to generate sensor mask; create_x should return (2, num_sensors)
            x = create_x(T_input=inputs[0, 0], observed_fraction=observed_fraction, domain_fraction=domain_fraction)

        # Precompute global min/max over all inputs (all samples and all time steps)
        u_global_max = torch.max(inputs)
        u_global_min = torch.min(inputs)
        # (Optional) If you want to normalize outputs based on outputs themselves, 
        # you could compute v_global_max/min similarly:
        # v_global_max = torch.max(outputs)
        # v_global_min = torch.min(outputs)

        for i in range(num_samples):
            # Convert normalized sensor coords x (in [0,1]) to integer row/col indices
            cols = (x[0] * (W - 1)).round().long()  # x[0] → columns
            rows = (x[1] * (H - 1)).round().long()  # x[1] → rows

            # Extract sensor readings for each of the seq_len time frames:
            # inputs[i] has shape (seq_len, H, W), so indexing gives (seq_len, num_sensors)
            u_seq = inputs[i, :, rows, cols]  # shape: (seq_len, num_sensors)

            # Normalize u_seq using global min/max from inputs
            u_seq = (u_seq - u_global_min) / (u_global_max - u_global_min)

            # Output field v (single frame) has shape (H, W)
            v_field = outputs[i]
            # Normalize v_field using the same global min/max from inputs
            v_field = (v_field - u_global_min) / (u_global_max - u_global_min)
            # If you prefer using v's own min/max, uncomment below instead:
            # v_field = (v_field - v_global_min) / (v_global_max - v_global_min)

            # (Optional) plotting/debugging
            plotting = False
            if plotting:
                fig, ax = plt.subplots(1, 3, figsize=(18, 5))

                # Plot the last snapshot in the sequence at sensor locations
                last_frame = inputs[i, -1].cpu()
                scatter_1 = ax[0].scatter(
                    x[0].cpu(),
                    x[1].cpu(),
                    c=u_seq[-1].cpu(),
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )
                ax[0].set_aspect("equal")
                cbar1 = fig.colorbar(scatter_1, ax=ax[0])
                cbar1.set_label("u (last frame)")

                # Plot the full output field v_field
                im_1 = ax[1].imshow(
                    v_field.cpu(),
                    cmap="viridis",
                    origin="lower",
                    vmin=0,
                    vmax=1,
                )
                ax[1].set_aspect("equal")
                cbar2 = fig.colorbar(im_1, ax=ax[1])
                cbar2.set_label("v (normalized)")

                # Plot one of the intermediate frames from u_seq (e.g., first)
                scatter_2 = ax[2].scatter(
                    x[0].cpu(),
                    x[1].cpu(),
                    c=u_seq[0].cpu(),
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )
                ax[2].set_aspect("equal")
                cbar3 = fig.colorbar(scatter_2, ax=ax[2])
                cbar3.set_label("u (first frame)")

                plt.tight_layout()
                plt.savefig(f"./debug_sample_{i}.png")
                plt.show()
                plt.close()

            u_data.append(u_seq)      # (seq_len, num_sensors)
            v_data.append(v_field)    # (H, W)

        # Stack into tensors:
        # u_stack: (num_samples, seq_len, num_sensors)
        u_stack = torch.stack(u_data)  
        # v_stack: (num_samples, H, W) → add channel dim → (num_samples, 1, H, W)
        v_stack = torch.stack(v_data).unsqueeze(1)

        # Expand x and y to have batch dimension:
        # x: (2, num_sensors) → (1, 2, num_sensors) → (num_samples, 2, num_sensors)
        x_batch = x.unsqueeze(0).expand(num_samples, *x.shape)
        # y: (2, H, W) → (1, 2, H, W) → (num_samples, 2, H, W)
        y_batch = y.unsqueeze(0).expand(num_samples, *y.shape)

        # Call parent constructor: u has shape (N, C_in, M), v has shape (N, C_out, H, W)
        super().__init__(u=u_stack, v=v_stack, x=x_batch, y=y_batch)

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
    num_simulations = 1000
    nt = 5000
    t0 = nt - 1
    d_min = 0.1
    d_max = 0.3
    n_sensor_timestamps = 20
    start_y = 4
    end_y = 38  
    start_x = 24
    end_x = 98
    dt = 1e-4  
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    noise_amplitude = 0.0
    n_blobs = 200
    radius = 5
    quadratic_alpha = False  # If True, use quadratic alpha for diffusivity
    snapshot_simulation = True  # If True, generate a snapshot simulation dataset
    random_t = True  # If True, generate random snapshots; if False, use the last snapshot
    
    if quadratic_alpha:
        # Use d_min and d_max for din and dout respectively as per current variable names
        snapshot_prefix = "snapshot_" if snapshot_simulation else ""
        save_path_simulation = os.path.join(script_dir, "datasets", "simulation", \
                                    f"{snapshot_prefix}{timestamp}_simulation_n{num_simulations}_nt{nt}_nx{nx}_ny{ny}" \
                                    f"_dt{dt}_din{d_min}_ntsensor{n_sensor_timestamps}_dout{d_max}_sy{start_y}_ey{end_y}_sx{start_x}_ex{end_x}_random{random_t}.pt")
        D = torch.full((ny, nx), d_max, device=device)
        D[start_y:end_y, start_x:end_x] = d_min
    else:
        snapshot_prefix = "snapshot_" if snapshot_simulation else ""
        save_path_simulation = os.path.join(script_dir, "datasets", "simulation", 
                                    f"{snapshot_prefix}{timestamp}_simulation_n{num_simulations}_nt{nt}_nx{nx}_ny{ny}"
                                    f"_dt{dt}_dmin{d_min}_ntsensor{n_sensor_timestamps}_dmax{d_max}_nblobs{n_blobs}_radius{radius}_random{random_t}.pt")
        D = create_blob_diffusivity(ny=ny, nx=nx, d_min=d_min, d_max=d_max, n_blobs=n_blobs, radius=radius, device=device)

    print(save_path_simulation)
    if not os.path.exists(os.path.dirname(save_path_simulation)):
        print("no such path")
        exit()
    
    plt.figure(figsize=(6, 5))
    plt.imshow(D.cpu().numpy(), cmap='viridis', origin='lower')
    plt.colorbar(label='Diffusion Coefficient')
    plt.savefig("diffusion_coefficient.png")
    plt.show()

    if not snapshot_simulation:
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
    else:
        simulation_dataset = SnapshotsSimulationDataset(
            num_simulations=num_simulations,
            random_t=random_t,  # Set to True to generate random snapshots
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            D=D,
            nt=nt,
            dt=dt,
            n_sensor_timestamps=n_sensor_timestamps,
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