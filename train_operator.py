import torch
import matplotlib.pyplot as plt
from continuiti.operators import DeepCatOperator
from continuiti.data.utility import split
import random 
import os
from datetime import datetime
from torch.utils.data import DataLoader
from continuiti.operators.shape import OperatorShapes, TensorShape

from torch.utils.tensorboard import SummaryWriter
from dataset import OperatorFieldMappingDataset
from continuiti.trainer.scheduler import LinearLRScheduler
from continuiti.trainer.callbacks import PrintTrainingLoss, Logs
import math
from lstm_model import *
from visualization import visualize_predictions
from trainer import Trainer
# Initialize TensorBoard writer
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script folder
    os.chdir(script_dir)  # Set script directory as working directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    ############### CREATE OPERATOR DATASET ###############
    num_samples = 2000
    observed_fraction = 0.0004
    domain_fraction = 1
    simulation_file = "snapshot_0613_101159_simulation_n100_nt5000_nx100_ny100_dt0.0001_dmin0.1_ntsensor20_dmax0.3_nblobs200_radius5_randomTrue.pt"
    simulation_file_path = os.path.join(script_dir, "datasets", "simulation", simulation_file)
    simulation_file = simulation_file.replace(".pt", "")
    
    # Define a 4x4 grid of points, taking centers of subdivisions
    num_points_x = 2
    num_points_y = 2
    
    # Generate coordinates that are centers of subdivisions
    # For an interval [0, 1] and n points, coordinates are (i + 0.5) / n
    x_coords = (torch.arange(num_points_x) + 0.5) / num_points_x
    y_coords = (torch.arange(num_points_y) + 0.5) / num_points_y

    # Create a grid of points
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij') # 'ij' indexing for (H, W) style grid
    # Flatten and stack to get coordinates in the shape (num_total_points, 2)
    # where each row is [x, y]
    sensor_coordinates = torch.stack([xx.flatten(), yy.flatten()], dim=1).transpose(0, 1)

    print("sensor coordinates shape:", sensor_coordinates.shape) # Should be torch.Size([2, 16]) for a 4x4 grid
    print("sensor coordinates (first 5 columns):", sensor_coordinates[:, :5])

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

    # visualize_dataset(dataset, n=10)

    if len(dataset) > 1:
        train_dataset, test_dataset = split(dataset, 0.8)
    else:
        train_dataset = dataset

    #### logging and saving ####
    log_tensorboard = True
    save_model = False
    ############################

    # Define hyperparameters
    epochs = 200
    trunk_depth = 16
    branch_depth = 16
    trunk_width = 48
    branch_width = 48
    batch_size = 32
    weight_decay = 0
    lstm = True


    num_sensors = sensor_coordinates.shape[1]
    new_u_shape = TensorShape(dim=1, size=([num_sensors]))
    operator_shapes = OperatorShapes(
        x=dataset.shapes.x,
        u=new_u_shape,
        y=dataset.shapes.y,
        v=dataset.shapes.v,
    )

    # Instantiate the operator using those variables
    operator = DeepCatOperator(
        shapes=operator_shapes, 
        device=device, 
        trunk_depth=trunk_depth, 
        branch_depth=branch_depth, 
        trunk_width=trunk_width, 
        branch_width=branch_width
    )

    operator = operator.to(device)
    
    lstm_network = None
    if lstm:
        lstm_network = LSTM().to(device)
        optimizer = torch.optim.Adam(
            list(operator.parameters()) + list(lstm_network.parameters()),
            lr=1e-3,
            weight_decay=weight_decay
        )
    else:
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
        'scheduler': scheduler.__class__.__name__,
        'batch_size': batch_size,
        'lstm': lstm,
    }

    # Create TensorBoard logger
    log_dir = f"runs/{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    mse_loss = torch.nn.MSELoss()
    hparam_text = "\n".join(f"{key}: {value}" for key, value in hparams.items())   

    if log_tensorboard:
        writer.add_text("Hyperparameters", hparam_text)

    # -------------------------------------------------------------
    #  THREE-STAGE TRAINING PIPELINE
    #  Stage 1 : train Operator only
    #  Stage 2 : train LSTM only (Operator frozen)
    #  Stage 3 : fine-tune Operator + LSTM end-to-end
    # -------------------------------------------------------------

    # ---------------------------
    #  Stage 1 – Operator only
    # ---------------------------
    print("\n====================  Stage 1 : training Operator only  ====================")
    lstm_network_stage1 = None                        # no LSTM in Stage 1
    optimizer_stage1 = torch.optim.Adam(operator.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler_stage1 = LinearLRScheduler(optimizer=optimizer_stage1, max_epochs=epochs)
    writer_stage1 = SummaryWriter(log_dir=os.path.join(log_dir, "stage1")) if log_tensorboard else None

    trainer_stage1 = Trainer(
        operator=operator,
        lstm_network=lstm_network_stage1,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        loss_fn=mse_loss,
        device=device,
        writer=writer_stage1,
        use_lstm=False,            # LSTM not used
    )
    trainer_stage1.train(train_loader, val_loader, epochs)

    # ---------------------------
    #  Stage 2 – LSTM only
    # ---------------------------
    print("\n====================  Stage 2 : training LSTM only  ====================")
    # Freeze Operator weights
    for p in operator.parameters():
        p.requires_grad = False
    operator.eval()

    # Instantiate LSTM (fresh) if it does not yet exist
    lstm_network = LSTM().to(device)

    optimizer_stage2 = torch.optim.Adam(lstm_network.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler_stage2 = LinearLRScheduler(optimizer=optimizer_stage2, max_epochs=epochs)
    writer_stage2 = SummaryWriter(log_dir=os.path.join(log_dir, "stage2")) if log_tensorboard else None

    trainer_stage2 = Trainer(
        operator=operator,
        lstm_network=lstm_network,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        loss_fn=mse_loss,
        device=device,
        writer=writer_stage2,
        use_lstm=True,             # now we use the LSTM
    )
    trainer_stage2.train(train_loader, val_loader, epochs)

    # ---------------------------
    #  Stage 3 – fine-tune all
    # ---------------------------
    print("\n====================  Stage 3 : end-to-end fine-tuning  ====================")
    # Un-freeze Operator parameters
    for p in operator.parameters():
        p.requires_grad = True
    operator.train()

    # Ensure LSTM weights are also trainable
    for p in lstm_network.parameters():
        p.requires_grad = True

    optimizer_stage3 = torch.optim.Adam([
        {"params": operator.parameters(), "lr": 1e-4},
        {"params": lstm_network.parameters(), "lr": 1e-5},
    ], weight_decay=weight_decay)
    scheduler_stage3 = LinearLRScheduler(optimizer=optimizer_stage3, max_epochs=epochs)

    writer_stage3 = SummaryWriter(log_dir=os.path.join(log_dir, "stage3")) if log_tensorboard else None

    trainer_stage3 = Trainer(
        operator=operator,
        lstm_network=lstm_network,
        optimizer=optimizer_stage3,
        scheduler=scheduler_stage3,
        loss_fn=mse_loss,
        device=device,
        writer=writer_stage3,
        use_lstm=True,
    )
    trainer_stage3.train(train_loader, val_loader, epochs)

    # -------------------------------------------------------------
    #  VISUALISATION AFTER TRAINING
    # -------------------------------------------------------------
    visualize_predictions(operator, lstm_network, train_dataset, num_samples=5, mode="train", device=device, log_dir=log_dir)
    visualize_predictions(operator, lstm_network, test_dataset, num_samples=5, mode="test", device=device, log_dir=log_dir)


if __name__ == "__main__":
    main()
