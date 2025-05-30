import torch
import matplotlib.pyplot as plt
from continuiti.operators import DeepCatOperator, DeepONet
from continuiti.trainer import Trainer
from continuiti.trainer.callbacks import LearningCurve
from continuiti.data.utility import split
from Callbacks import ModelCheckpointCallback, TensorBoardLogger
import random 
import os
from datetime import datetime
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from dataset import OperatorTemperatureDataset, OperatorFieldMappingDataset
from continuiti.trainer.scheduler import LinearLRScheduler
from continuiti.trainer.callbacks import PrintTrainingLoss, Logs
import math

# Initialize TensorBoard writer
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")

def visualize_predictions(operator, test_dataset, num_samples, mode, device='cpu', save_folder="result", filename=None, log_dir=None):
    operator.eval()
    operator.to(device)

    if filename == None:
        timestamp = datetime.now().strftime("%m%d-%H%M%S")
        filename = f"{timestamp}_{mode}.png"

    num_samples = min(num_samples, len(test_dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        x,u,y,v = test_dataset[i]
        print("u shape pred", u.shape)
        print("x shape pred", x.shape)
        print("v shape pred", v.shape)
        print("y shape pred", y.shape)
        u_max = torch.max(u)
        u_min = torch.min(u)
        v_max = torch.max(v)
        v_min = torch.min(v)
        u = (u - u_min) / (u_max- u_min)
        v = (v - v_min) / (v_max - v_min)


        x = x.to(device).unsqueeze(0)
        u = u.to(device).unsqueeze(0)
        y = y.to(device).unsqueeze(0)

        with torch.no_grad():
            prediction = operator(x, u, y).cpu().squeeze()
        
        # Select row of subplots
        x = x.squeeze().cpu()
        u = u.squeeze().cpu()
        y = y.squeeze().cpu()
        v = v.squeeze().cpu()
        prediction = prediction.squeeze().cpu().numpy()

        x_min = y[0].min()
        x_max = y[0].max()
        print("prediction: u", u)
        
        ax1, ax2, ax3 = axes[i] if num_samples > 1 else axes  # Handle single sample case

        ax1.set_title("Input")
        im1 = ax1.scatter(x[0,:], x[1,:], c=u, cmap="viridis", vmin=0, vmax=1)
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
        print("u shape", u_sample.shape)
        print("x shape", x_sample.shape)
        print("v shape", v_sample.shape)
        print("y shape", y_sample.shape)

        # Remove channel dimension from v and get dimensions
        print("u_sample", u_sample)
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


def train_one_epoch(current_epoch, tb_writer): 
    pass


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script folder
    os.chdir(script_dir)  # Set script directory as working directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ################ LOAD OPERATOR DATASET ############
    # data_file_name = "operator_m1000_oberserved0.1_domain0.5_simulation_n5000_t0299_t0.030_nx10_ny20.pt"
    # save_path = os.path.join(script_dir, "datasets", data_file_name)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    # # Create dataset
    # if os.path.exists(save_path):
    #     dataset = torch.load(save_path, weights_only=False)
    #     print(f" ############## DATASET: {data_file_name}, SIZE: {len(dataset)} ##################")
    #     print(dataset.shapes)
    ############################################

    ############### CREATE OPERATOR DATASET ###############
    num_samples = 3000
    observed_fraction = 0.0004
    domain_fraction = 1
    simulation_file = "0424_153319_simulation_n10000_t00.030_t0.030_nx100_ny100_din0.1_dout0.3_sy4_ey38_sx24_ex98.pt"
    simulation_file_path = os.path.join(script_dir, "datasets", "simulation", simulation_file)
    simulation_file = simulation_file.replace(".pt", "")
    sensor_coordinates = torch.tensor([[0.2, 0.1], [0.4, 0.1], [0.6, 0.1], [0.8, 0.1]]).transpose(0, 1)
    print("sensor coordinates", sensor_coordinates)
    dataset = OperatorFieldMappingDataset(
        num_samples=num_samples,
        sensor_coordinates=sensor_coordinates,
        observed_fraction=observed_fraction, 
        domain_fraction=domain_fraction,
        simulation_file_path=simulation_file_path,
        save_path=None
    )

    _, H, W = dataset.v[0].shape
    n_observed_points = observed_fraction * H * W * domain_fraction
    print("observed points:", n_observed_points)
    data_file_name = f"operator_m{num_samples}_oberserved{observed_fraction}_npoints{n_observed_points}_domain{domain_fraction}_{simulation_file}.pt"
    ######################################################
    
    print(f"Dataset size: {len(dataset)} samples")

    visualize_dataset(dataset, n=5)

    if len(dataset) > 1:
        train_dataset, test_dataset = split(dataset, 0.8)
    else:
        train_dataset = dataset

    #### logging and saving ####
    log_tensorboard = True
    save_model = False
    ############################

    # Define hyperparameters
    epochs = 4000
    trunk_depth = 16
    branch_depth = 16
    trunk_width = 32
    branch_width = 32
    batch_size = 32
    weight_decay = 0
    # Instantiate the operator using those variables
    operator = DeepCatOperator(
        shapes=dataset.shapes, 
        device=device, 
        trunk_depth=trunk_depth, 
        branch_depth=branch_depth, 
        trunk_width=trunk_width, 
        branch_width=branch_width
    )

    operator = operator.to(device)
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = LinearLRScheduler(optimizer=optimizer, max_epochs=epochs)

    # Count parameters
    total_params = sum(p.numel() for p in operator.parameters())
    print("total param: ", total_params)

    # Create hyperparameter dict
    hparams = {
        'trunk_depth': trunk_depth,
        'branch_depth': branch_depth,
        'trunk_width': trunk_width,
        'branch_width': branch_width,
        'epochs': epochs,
        'dataset_size': len(dataset),
        'model_type': type(operator).__name__,
        'total_params': total_params,
        'dataset_name': data_file_name,
        'weight_decay': weight_decay,
        'scheduler': scheduler.__class__.__name__
    }

    # Create TensorBoard logger
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    log_dir = f"runs/{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()

    hparam_text = "\n".join(f"{key}: {value}" for key, value in hparams.items())   

    if log_tensorboard:
        writer.add_text("Hyperparameters", hparam_text)

    steps = math.ceil(len(train_dataset) / batch_size)
    print_loss_callback = PrintTrainingLoss(epochs, steps)

    # Training
    for epoch in range(epochs):
        loss_train = 0.0
        operator.train()
        
        logs = Logs(epoch=epoch + 1, step=0, loss_train=None, loss_test=None)
        for x, u, y, v in train_loader:
            operator.zero_grad()
            x = x.to(device)
            u = u.to(device)
            y = y.to(device)
            v = v.to(device)

            pred = operator(x, u, y)
            pred = pred.reshape(pred.shape)
            loss = mse_loss(pred, v)
            loss.backward()
            optimizer.step()
            loss_train += loss.detach().item()
            logs.step += 1
            logs.loss_train = loss_train / logs.step
            
            print_loss_callback.step(logs)

        loss_train /= len(train_loader)

        if log_tensorboard:
            writer.add_scalar("Loss/Train", loss_train, epoch)

        if test_dataset is not None:
            operator.eval()
            loss_eval = 0.0
            with torch.no_grad():
                for x, u, y, v in val_loader:
                    x = x.to(device)
                    u = u.to(device)
                    y = y.to(device)
                    v = v.to(device)
                    pred = operator(x, u, y)
                    pred = pred.reshape(v.shape)

                    loss = mse_loss(pred, v)
                    loss_eval += loss.detach().item()

            loss_eval /= len(val_loader)
            logs.loss_test = loss_eval
            print_loss_callback(logs)

            if log_tensorboard:
                writer.add_scalar("Loss/Eval", loss_eval, epoch)

            scheduler(logs)


    # tensoorboard_callback = TensorBoardLogger(log_dir=log_dir)
    # callbacks = [tensoorboard_callback]
    # trainer = Trainer(operator=operator)
    # trainer.fit(epochs=epochs, callbacks=callbacks, dataset=train_dataset, test_dataset=test_dataset)

    visualize_predictions(operator, train_dataset, num_samples=10, mode="train", device=device, log_dir=log_dir)
    visualize_predictions(operator, test_dataset, num_samples=10, mode="test", device=device, log_dir=log_dir)


if __name__ == "__main__":
    main()
