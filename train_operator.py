import torch
import matplotlib.pyplot as plt
from continuiti.operators import DeepCatOperator
from continuiti.trainer import Trainer
from continuiti.trainer.callbacks import LearningCurve
from continuiti.data.utility import split
from Callbacks import ModelCheckpointCallback, TensorBoardLogger
import random 
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from continuiti.trainer.scheduler import LinearLRScheduler

# Initialize TensorBoard writer
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
writer = SummaryWriter(f"runs/{timestamp}")


def visualize_predictions(operator, test_dataset, num_samples, device='cpu', save_folder="result", filename=None, log_dir=None):
    operator.eval()
    operator.to(device)

    if filename == None:
        timestamp = datetime.now().strftime("%m%d-%H%M%S")
        filename = f"{timestamp}.png"

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(min(num_samples, len(test_dataset))):
        x, u, y, v = test_dataset[i] 

        y_size = v.shape[1]
        grid_x, grid_y = torch.meshgrid(torch.arange(0, y_size, dtype=torch.float32), torch.arange(0, y_size, dtype=torch.float32))
        y_field = torch.stack([grid_x, grid_y])
        x = x.to(device).unsqueeze(0)
        u = u.to(device).unsqueeze(0)
        y_field = y_field.to(device).unsqueeze(0)

        with torch.no_grad():
            prediction = operator(x, u, y_field).cpu().squeeze()
        
        # Select row of subplots

        x = x.cpu().squeeze(0).numpy()
        ax1, ax2, ax3 = axes[i] if num_samples > 1 else axes  # Handle single sample case

        ax1.set_title("Partial Input")
        im1 = ax1.scatter(x[0,:], x[1,:], s=3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect("equal")
        fig.colorbar(im1, ax=ax1)

        ax2.set_title("Ground Truth")
        im2 = ax2.imshow(v.squeeze().cpu().numpy(), cmap='viridis', origin="lower")
        ax2.set_aspect("equal")
        ax2.set_xlim(0, 99)
        ax2.set_ylim(0, 99)
        fig.colorbar(im2, ax=ax2)

        ax3.set_title("Prediction")
        im3 = ax3.imshow(prediction, cmap='viridis', origin="lower")
        ax3.set_aspect("equal")
        ax3.set_xlim(0, 99)
        ax3.set_ylim(0, 99)
        fig.colorbar(im3, ax=ax3)

    plt.tight_layout()

    # Save to file
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    
    # Log figure to TensorBoard
    if log_dir is not None:
        viz_writer = SummaryWriter()
        viz_writer.add_figure('Predictions', fig, close=False)
        print("Figure added to TensorBoard")
        viz_writer.close()
    
    plt.show()

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
        x_sample = dataset.x[idx]  # shape: (2, num_observed)
        v_sample = dataset.v[idx]  # shape: (1, H, W)
        
        # Remove channel dimension from v and get dimensions
        v_field = v_sample.squeeze(0).cpu().numpy()  # shape: (H, W)
        H, W = v_field.shape
        
        # Get observed coordinates from x_sample
        x_coords = x_sample[0].cpu().numpy()  # column coordinates
        y_coords = x_sample[1].cpu().numpy()  # row coordinates
        
        # If coordinates are normalized to [0, 1], scale them to image dimensions
        if x_coords.max() <= 1.1 and y_coords.max() <= 1.1:
            x_coords_plot = x_coords * (W - 1)
            y_coords_plot = y_coords * (H - 1)
        else:
            x_coords_plot = x_coords
            y_coords_plot = y_coords
        
        # --- Left subplot: Observed input (x) ---
        ax_input = axes[i, 0]
        ax_input.scatter(x_coords_plot, y_coords_plot, color='blue', s=3)
        ax_input.set_xlim(0, W - 1)
        ax_input.set_ylim(0, H - 1)
        ax_input.set_aspect("equal")
        ax_input.set_title(f"Sample {idx}: Observed Input")
        ax_input.set_xlabel("X index")
        ax_input.set_ylabel("Y index")
        # Optionally, invert the y-axis to match image coordinate systems
        
        # --- Right subplot: Full temperature field (v) ---
        ax_field = axes[i, 1]
        im = ax_field.imshow(v_field, cmap='viridis', origin='lower')
        ax_field.set_title(f"Sample {idx}: Full Temperature Field")
        ax_field.set_xlabel("X index")
        ax_field.set_ylabel("Y index")
        fig.colorbar(im, ax=ax_field)
    
    plt.tight_layout()
    plt.show()


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script folder
    os.chdir(script_dir)  # Set script directory as working directory
    data_file_name = "operator_dataset_1000.pt"
    save_path = os.path.join(script_dir, "datasets", data_file_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    if os.path.exists(save_path):
        dataset = torch.load(save_path, weights_only=False)
        print(f" ############## DATASET: {data_file_name}, SIZE: {len(dataset)} ##################")
        print(dataset.shapes)

    # visualize_dataset(dataset, n=5)
    train_dataset, test_dataset = split(dataset, 0.8)


    #### logging and saving ####

    log_tensorboard = True
    save_model = True

    ############################


    # Define hyperparameters
    trunk_depth = 8
    epochs = 5
    
    operator = DeepCatOperator(shapes=dataset.shapes, trunk_depth=trunk_depth, device=device)
    total_params = sum(p.numel() for p in operator.parameters())
    optimizer = torch.optim.Adam(operator.parameters(), lr=0.001)

    model_type = type(operator).__name__
    # Collect hyperparameters in a dictionary
    hparams = {
        'trunk_depth': trunk_depth,
        'epochs': epochs,
        'dataset_size': len(dataset),
        'model_type': model_type,
        'total_params': total_params
    }
    
    # Create TensorBoard logger
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    log_dir = f"runs/{timestamp}"

    if log_tensorboard:
        tb_logger = TensorBoardLogger(log_dir=log_dir, log_weights=True, hparams=hparams) 
    else:
        tb_logger = None

    model_name = f"{model_type}_{timestamp}"

    scheduler = LinearLRScheduler(optimizer=optimizer, max_epochs=epochs)
    ##################### Uncomment for load model #######################

    # checkpoint_path = os.path.join(script_dir, "saved_models", f"DeepCatOperator_04-03_10-01-12_final.pt")  # use your actual saved filename
    # checkpoint = torch.load(checkpoint_path, map_location=device)

    # print("checkpoint", checkpoint.keys())
    # operator.load_state_dict(checkpoint['model_state_dict'])
    # operator.to(device)  # Ensure the model is on the correct device

    # optimizer = torch.optim.Adam(operator.parameters(), lr=0.001)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # if 'scheduler_state_dict' in checkpoint:
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # print("Operator and optimizer state loaded successfully.")

    #######################################################################


    # ################## Uncomment for train model #########################

    trainer = Trainer(operator=operator, optimizer=optimizer, device=device)

    if save_model:
        checkpoint_callback = ModelCheckpointCallback(
            operator,
            trainer.optimizer,
            save_dir=os.path.join(script_dir, "saved_models", model_name),
            save_interval=250,
            scheduler=scheduler
        ) 
    else:
        None
    
    trainer.fit(train_dataset, test_dataset=test_dataset, callbacks=[LearningCurve(), tb_logger, checkpoint_callback], lr_scheduler=scheduler, epochs=epochs)

    model_save_path = os.path.join(script_dir, "saved_models", f"{model_name}_final.pt")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    final_checkpoint = {
        'epoch': epochs,  # or use the last completed epoch if different
        'model_state_dict': operator.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }

    final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(final_checkpoint, model_save_path)
    print(f"Checkpoint saved to {model_save_path}")

    # ######################################################################

    visualize_predictions(operator, train_dataset, num_samples=10, device=device, log_dir=log_dir)

if __name__ == "__main__":
    main()
