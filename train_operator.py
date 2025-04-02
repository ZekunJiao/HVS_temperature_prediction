import torch
import matplotlib.pyplot as plt
from continuiti.operators import DeepCatOperator
from continuiti.trainer import Trainer
from continuiti.trainer.callbacks import LearningCurve
from continuiti.data.utility import split
from dataset import OperatorTemperatureDataset
from TensorBoardlogger import TensorBoardLogger

import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import wandb

# Initialize TensorBoard writer
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
writer = SummaryWriter(f"runs/{timestamp}")


def visualize_predictions(operator, test_dataset, num_samples, writer, device='cpu', save_folder="result",  filename=None):
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
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_aspect("equal")
        fig.colorbar(im1, ax=ax1)

        ax2.set_title("Ground Truth")
        im2 = ax2.imshow(v.squeeze().cpu().numpy(), cmap='viridis', origin="lower")
        ax2.set_aspect("equal")
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        fig.colorbar(im2, ax=ax2)

        ax3.set_title("Prediction")
        im3 = ax3.imshow(prediction, cmap='viridis', origin="lower")
        ax3.set_aspect("equal")
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 100)
        fig.colorbar(im3, ax=ax3)

    plt.tight_layout()

    # Save to file
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    
    # Log figure to TensorBoard
    if writer is not None:
        writer.add_figure('Predictions', fig, close=False)
        print("Figure added to TensorBoard")
    
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


    train_dataset, test_dataset = split(dataset, 0.8)

    # Define hyperparameters
    trunk_depth = 8
    epochs = 1000
    
    operator = DeepCatOperator(shapes=dataset.shapes, trunk_depth=trunk_depth, device=device)
    trainer = Trainer(operator, device=device)
    
    # Collect hyperparameters in a dictionary
    hparams = {
        'trunk_depth': trunk_depth,
        'epochs': epochs,
        'dataset_size': len(dataset),
        'model_type': 'DeepCatOperator'
    }
    
    # Create TensorBoard logger
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    log_dir = f"runs/{timestamp}"
    tb_logger = TensorBoardLogger(log_dir=log_dir, log_weights=True, hparams=hparams)
    
    trainer.fit(train_dataset, test_dataset=train_dataset, callbacks=[LearningCurve(), tb_logger], epochs=epochs)
    print("vizualizing")

    # Create a new writer for visualizations using the same log directory
    viz_writer = SummaryWriter(log_dir)
    visualize_predictions(operator, train_dataset, num_samples=10, writer=viz_writer, device=device)
    viz_writer.close()  # Close the visualization writer when done

if __name__ == "__main__":
    main()
