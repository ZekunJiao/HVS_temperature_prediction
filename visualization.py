import torch
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset import OperatorFieldMappingDataset

def visualize_predictions(operator, lstm_network, test_dataset, num_samples, mode, device='cpu', save_folder="result", filename=None, log_dir=None):
    operator.eval()
    operator.to(device)

    if filename == None:
        timestamp = datetime.now().strftime("%m%d-%H%M%S")
        filename = f"{timestamp}_{mode}.png"

    num_samples = min(num_samples, len(test_dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        x,u_raw,y,v = test_dataset[i]

        x = x.to(device).unsqueeze(0)
        u_raw = u_raw.to(device).unsqueeze(0)
        print("u_raw shape: ", u_raw.shape)
        y = y.to(device).unsqueeze(0)

        if lstm_network is not None:
            with torch.no_grad():
                print("using lstm")
                u_pred = lstm_network(u_raw)  # Apply LSTM to condense sensor data

                # Compare LSTM output with actual last sensor values
                actual_last_u = u_raw[:, -1]
                predicted_last_u = u_pred[:, -1, :]
                print("\n--- LSTM Visualization Debug ---")
                print(f"Sample {i}:")
                print("  Actual sensor values (last sequence step):\n", actual_last_u.squeeze().cpu().numpy())
                print("  LSTM predicted values (last sequence step):\n", predicted_last_u.squeeze().cpu().numpy())
                print("---------------------------------\n")

                u = predicted_last_u
                u = torch.unsqueeze(u, 1)  # Reshape to match expected input shape
                prediction = operator(x, u, y).cpu().squeeze()
        else:
            with torch.no_grad():
                print("not using lstm") 
                u = u_raw[:, -1]
                prediction = operator(x, u, y).cpu().squeeze()
        
        # Select row of subplots
        x = x.squeeze().cpu()
        u_raw = u_raw.squeeze().cpu()
        y = y.squeeze().cpu()
        v = v.squeeze().cpu()
        prediction = prediction.squeeze().cpu().numpy()

        ax1, ax2, ax3 = axes[i] if num_samples > 1 else axes  # Handle single sample case

        ax1.set_title("Input")
        im1 = ax1.scatter(x[0,:], x[1,:], c=u_raw[-1], cmap="viridis", vmin=0, vmax=1)
        ax1.set_xlim(y[0].min(), y[0].max())
        ax1.set_ylim(y[1].min(), y[1].max())
        ax1.set_aspect("equal")
        fig.colorbar(im1, ax=ax1)

        ax2.set_title("Ground Truth")
        im2 = ax2.scatter(y[1], y[0], c=v, cmap='viridis', vmin=0, vmax=1)
        ax2.set_aspect("equal")
        fig.colorbar(im2, ax=ax2)

        ax3.set_title("Prediction")
        im3 = ax3.scatter(y[1], y[0], c=prediction, cmap='viridis', vmin=0, vmax=1)
        ax3.set_aspect("equal")
        fig.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.suptitle(f"{mode}")

    # Save to file
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    
    # Log figure to TensorBoard
    if log_dir is not None:
        viz_writer = SummaryWriter(log_dir=log_dir)
        viz_writer.add_figure(f'{mode}_{len(test_dataset)}', fig, close=False)
        print("Figure added to TensorBoard")
        viz_writer.close()
    
    # plt.show()

def visualize_dataset(dataset, n=1):
    """
    Visualize n samples from the dataset for a sanity check.
    
    For each sample, two subplots are shown in one row:
      - Left: Observed input (x) as a scatter plot.
      - Right: Full temperature field (v) as a heatmap.
    
    Args:
        dataset: The dataset instance which is expected to have attributes x and v.
                 - x: Observed coordinate input of shape (num_samples, 2, num_observed)
                 - v: Full field values, shape (num_samples, 1, H, W)
        n (int): Number of samples to visualize, randomly sampled from the dataset.
    """
    # Randomly sample n unique indices from the dataset
    n = min(n, len(dataset))
    
    if n <= 0:
        raise ValueError("Number of samples to visualize must be positive")

    indices = random.sample(range(len(dataset)), n)
    
    n_samples = len(indices)
    n_cols = 2  # one column for x (observed points) and one for v (full field)
    
    # Create a figure with one row per sample
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(6 * n_cols, 4 * n_samples))
    
    # If only one sample is provided, ensure axes is 2D for consistency
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Extract sample data
        u_sample = dataset.u[idx]
        x_sample = dataset.x[idx]  # shape: (2, num_observed)
        v_sample = dataset.v[idx]  # shape: (1, H, W)
        y_sample = dataset.y[idx]  # shape: (1, H, W)

        # Remove channel dimension from v and get dimensions
        y_sample = y_sample.squeeze(0)  # shape: (H, W)
        # Get observed coordinates from x_sample
        x_cols = x_sample[0].cpu().numpy()  # column coordinates
        x_rows = x_sample[1].cpu().numpy()  # row coordinates
        
        # If coordinates are normalized to [0, 1], scale them to image dimensions
        y_rows = y_sample[0].cpu().numpy()  # column coordinates
        y_cols = y_sample[1].cpu().numpy()  # row coordinates

        # --- Left subplot: Observed input (x) ---
        ax_input = axes[i, 0]
        scatter_1 = ax_input.scatter(x_cols, x_rows, c=u_sample.squeeze().cpu(), cmap='viridis', vmin=0, vmax=1)
        ax_input.set_xlim(y_cols.min(), y_cols.max())
        ax_input.set_ylim(y_rows.min(), y_rows.max())
        ax_input.set_aspect("equal")
        ax_input.set_title(f"Sample {idx}: U")
        fig.colorbar(scatter_1, ax=ax_input)
        # Optionally, invert the y-axis to match image coordinate systems

        ax_y_sample = axes[i, 1]
        scatter_2 = ax_y_sample.scatter(y_cols, y_rows, c=v_sample.squeeze().cpu(), cmap="viridis", vmin=0, vmax=1)
        ax_y_sample.set_aspect("equal")
        ax_y_sample.set_title(f"Sample {idx}: V")

        cbar2 = fig.colorbar(scatter_2, ax=axes[i,1])
        cbar2.set_label("v")
    
    plt.tight_layout()
    plt.suptitle(f"Dataset Visualization: {n_samples} samples")
    plt.savefig("dataset_visualization.png")
    plt.show() 

# ----------------------------------------------------------------------------------
# Autocorrelation plotting utility
# ----------------------------------------------------------------------------------

def plot_autocorrelation(series, max_lag: int = 50):
    """Plot the autocorrelation function R(k) up to ``max_lag`` for all sensors in a 2D array.

    Args:
        series (np.ndarray | torch.Tensor): 2-D data array (time, sensors) or 1-D (time,).
        max_lag (int, optional): Maximum lag to display. Defaults to ``50``.
    """
    # Convert to NumPy array if a torch tensor is provided
    if isinstance(series, torch.Tensor):
        series_np = series.detach().cpu().numpy()
    else:
        series_np = np.asarray(series)

    if series_np.ndim == 1:
        series_np = series_np[:, None]  # (time,) -> (time, 1)
    elif series_np.ndim != 2:
        raise ValueError("`series` must be 1-D or 2-D (time, sensors).")

    n_time, n_sensors = series_np.shape
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError("`max_lag` must be non-negative.")

    # Prepare figure
    import matplotlib.pyplot as plt
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=False, gridspec_kw={'height_ratios': [2, 1]})

    # --- Top: Sensor values over time ---
    for i in range(1):
        print("sensor value", series_np[:, i])
        ax_top.plot(np.arange(n_time), series_np[:, i], label=f"Sensor {i+1}")
    ax_top.set_ylabel("Sensor Value")
    ax_top.set_title("Sensor Values Over Time (all sensors)")
    ax_top.grid(True, linestyle="--", alpha=0.7)
    if n_sensors <= 10:
        ax_top.legend(loc="best", fontsize="small")

    # --- Bottom: Autocorrelation for each sensor ---
    for i in range(1):
        s = series_np[:, i]
        s_centered = s - s.mean()
        ac_full = np.correlate(s_centered, s_centered, mode="full")
        mid = len(ac_full) // 2
        max_lag_i = min(max_lag, len(s) - 1)
        r = ac_full[mid : mid + max_lag_i + 1]
        if r[0] != 0:
            r = r / r[0]
        ks = np.arange(len(r))
        ax_bot.plot(ks, r, label=f"Sensor {i+1}")
    ax_bot.set_xlabel("Lag k")
    ax_bot.set_ylabel("Autocorrelation R(k)")
    ax_bot.set_title(f"Autocorrelation (max lag = {max_lag})")
    ax_bot.grid(True, linestyle="--", alpha=0.7)
    if n_sensors <= 10:
        ax_bot.legend(loc="best", fontsize="small")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------- 

