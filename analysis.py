import os
import torch
from dataset import OperatorFieldMappingDataset
from visualization import plot_autocorrelation


def load_dataset():
    """Load and return an OperatorFieldMappingDataset with the same parameters as train_operator.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    num_samples = 2000
    observed_fraction = 0.0004
    domain_fraction = 1
    simulation_file = "snapshot_0608_164903_simulation_n20_nt5000_nx100_ny100_dt0.0001_dmin0.1_ntsensor20_dmax0.3_nblobs200_radius5_randomTrue.pt"
    simulation_file_path = os.path.join(script_dir, "datasets", "simulation", simulation_file)

    num_points_x = 2
    num_points_y = 2
    x_coords = (torch.arange(num_points_x) + 0.5) / num_points_x
    y_coords = (torch.arange(num_points_y) + 0.5) / num_points_y
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    sensor_coordinates = torch.stack([xx.flatten(), yy.flatten()], dim=1).transpose(0, 1)

    dataset = OperatorFieldMappingDataset(
        num_samples=num_samples,
        sensor_coordinates=sensor_coordinates,
        observed_fraction=observed_fraction,
        domain_fraction=domain_fraction,
        simulation_file_path=simulation_file_path,
        save_path=None
    )
    return dataset


def main():
    # -----------------------------------------------------------------
    #  Autocorrelation analysis on all sensor time-series
    # -----------------------------------------------------------------
    dataset = load_dataset()
    if dataset.u.ndim == 3 and dataset.u.shape[1] > 1:
        # Use all sensors: shape (num_samples, seq_len, num_sensors)
        # We'll concatenate all samples along time for a long time-series per sensor
        u = dataset.u  # (num_samples, seq_len, num_sensors)
        n_samples, seq_len, n_sensors = u.shape
        # Reshape to (n_samples * seq_len, n_sensors)
        data_series = u[0]
        try:
            plot_autocorrelation(data_series, max_lag=50)
        except Exception as e:
            print(f"Autocorrelation plot failed: {e}")
    else:
        print("Dataset does not contain a suitable time-series for autocorrelation analysis.")


if __name__ == "__main__":
    main() 