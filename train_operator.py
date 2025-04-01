import torch
import matplotlib.pyplot as plt
from continuiti.operators import DeepCatOperator
from continuiti.trainer import Trainer
from continuiti.trainer.callbacks import LearningCurve
from continuiti.data.utility import split
from dataset import OperatorTemperatureDataset

import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import wandb

# Initialize TensorBoard writer
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
writer = SummaryWriter(f"runs/{timestamp}")

# Initialize a wandb run with a project name and config

# def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device):
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, targets in train_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             for name, param in model.named_parameters():
#                 writer.add_histogram(f'Weights/{name}', param, epoch)
#                 if param.grad is not None:
#                     writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
#             running_loss += loss.item() * inputs.size(0)
#         train_loss = running_loss / len(train_loader.dataset)

#         model.eval()
#         test_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets in test_loader:
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 test_loss += loss.item() * inputs.size(0)
#         test_loss /= len(test_loader.dataset)

#         # Log losses to TensorBoard
#         writer.add_scalar("Loss/Train", train_loss, epoch)
#         writer.add_scalar("Loss/Test", test_loss, epoch)

#         print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")


def visualize_predictions(operator, test_dataset, num_samples, writer, device='cpu', save_folder="result",  filename=None):
    operator.eval()

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
        print(x.shape)
        print(u.shape)
        print(y_field.shape)
        print(y.shape)

        with torch.no_grad():
            prediction = operator(x, u, y_field).cpu().squeeze()
        
        print(prediction.shape)
        # Select row of subplots

        x = x.cpu().squeeze(0).numpy()
        ax1, ax2, ax3 = axes[i] if num_samples > 1 else axes  # Handle single sample case

        ax1.set_title("Partial Input")
        im1 = ax1.scatter(x[0,:], x[1,:], s=3)
        ax1.set_aspect("equal")
        fig.colorbar(im1, ax=ax1)

        ax2.set_title("Ground Truth")
        im2 = ax2.imshow(v.squeeze().cpu().numpy(), cmap='viridis', origin="lower")
        ax2.set_aspect("equal")
        fig.colorbar(im2, ax=ax2)

        ax3.set_title("Prediction")
        im3 = ax3.imshow(prediction, cmap='viridis', origin="lower")
        ax3.set_aspect("equal")
        fig.colorbar(im3, ax=ax3)

    plt.tight_layout()

    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
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

    operator = DeepCatOperator(shapes=dataset.shapes, trunk_depth=8, device=device)
    trainer = Trainer(operator, device=device)
    epochs = 1
    # trainer.fit(train_dataset, test_dataset=test_dataset, callbacks=[LearningCurve()], epochs=epochs)
    print("vizualizing")

    visualize_predictions(operator, test_dataset, num_samples=1, writer=writer, device=device)
    writer.close()

if __name__ == "__main__":
    main()
