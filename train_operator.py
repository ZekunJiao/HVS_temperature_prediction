import torch
import matplotlib.pyplot as plt
from continuiti.operators import DeepCatOperator
from continuiti.data.utility import split
import random 
import os
import shutil
import sys
from datetime import datetime
from torch.utils.data import DataLoader
from continuiti.operators.shape import OperatorShapes, TensorShape

from torch.utils.tensorboard import SummaryWriter
from dataset import OperatorFieldMappingDataset
from continuiti.trainer.scheduler import LinearLRScheduler
from continuiti.trainer.callbacks import PrintTrainingLoss, Logs
import math
from lstm_model import *
from visualization import visualize_predictions, visualize_dataset
from trainer import Trainer


def train_stage_1(operator, train_loader, val_loader, epochs, weight_decay, device, base_log_dir, log_tensorboard, loss_fn):
    """Trains only the DeepCatOperator (Stage 1)."""
    print("\n==================== Stage 1: Training Operator only ====================")

    # No LSTM in Stage 1
    lstm_network_stage1 = None

    optimizer_stage1 = torch.optim.Adam(operator.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler_stage1 = LinearLRScheduler(optimizer=optimizer_stage1, max_epochs=epochs)
    writer_stage1 = SummaryWriter(log_dir=os.path.join(base_log_dir, "stage1")) if log_tensorboard else None

    trainer_stage1 = Trainer(
        operator=operator,
        lstm_network=lstm_network_stage1,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        loss_fn=loss_fn,
        device=device,
        writer=writer_stage1,
        use_lstm=False,
    )
    trainer_stage1.train(train_loader, val_loader, epochs)


def train_stage_2(operator, lstm_network, train_loader, val_loader, epochs, weight_decay, device, base_log_dir, log_tensorboard, loss_fn):
    """Trains only the LSTM with a frozen Operator (Stage 2)."""
    print("\n==================== Stage 2: Training LSTM only ====================")
    # Freeze Operator weights
    for p in operator.parameters():
        p.requires_grad = False
    operator.eval()

    optimizer_stage2 = torch.optim.Adam(lstm_network.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler_stage2 = LinearLRScheduler(optimizer=optimizer_stage2, max_epochs=epochs)
    writer_stage2 = SummaryWriter(log_dir=os.path.join(base_log_dir, "stage2")) if log_tensorboard else None

    trainer_stage2 = Trainer(
        operator=operator,
        lstm_network=lstm_network,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        loss_fn=loss_fn,
        device=device,
        writer=writer_stage2,
        use_lstm=True,
    )
    trainer_stage2.train(train_loader, val_loader, epochs)


def train_stage_3(operator, lstm_network, train_loader, val_loader, epochs, weight_decay, device, base_log_dir, log_tensorboard, loss_fn):
    """Fine-tunes both Operator and LSTM end-to-end (Stage 3)."""
    print("\n==================== Stage 3: End-to-end fine-tuning ====================")
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

    writer_stage3 = SummaryWriter(log_dir=os.path.join(base_log_dir, "stage3")) if log_tensorboard else None

    trainer_stage3 = Trainer(
        operator=operator,
        lstm_network=lstm_network,
        optimizer=optimizer_stage3,
        scheduler=scheduler_stage3,
        loss_fn=loss_fn,
        device=device,
        writer=writer_stage3,
        use_lstm=True,
    )
    trainer_stage3.train(train_loader, val_loader, epochs)


def train_jointly(operator, lstm_network, train_loader, val_loader, epochs, weight_decay, device, base_log_dir, log_tensorboard, loss_fn, use_lstm_flag):
    """Trains both Operator and LSTM (if enabled) jointly from the start."""
    print("\n==================== Joint Training ====================")
    
    if use_lstm_flag and lstm_network:
        print("Training jointly with LSTM using different learning rates (Operator: 1e-3, LSTM: 1e-4).")
        params_to_optimize = [
            {"params": operator.parameters(), "lr": 1e-3},
            {"params": lstm_network.parameters(), "lr": 1e-4},
        ]
        optimizer = torch.optim.Adam(params_to_optimize, weight_decay=weight_decay)
    else:
        print("Training operator only (joint mode, no LSTM).")
        params_to_optimize = operator.parameters()
        optimizer = torch.optim.Adam(params_to_optimize, lr=1e-3, weight_decay=weight_decay)

    scheduler = LinearLRScheduler(optimizer=optimizer, max_epochs=epochs)
    
    writer = SummaryWriter(log_dir=os.path.join(base_log_dir, "joint")) if log_tensorboard else None
    
    trainer = Trainer(
        operator=operator,
        lstm_network=lstm_network,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        writer=writer,
        use_lstm=use_lstm_flag,
    )
    trainer.train(train_loader, val_loader, epochs)


# Initialize TensorBoard writer
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")


def main():
    # --- Basic Setup ---
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset Creation ---
    num_samples = 2000
    observed_fraction = 0.0004
    domain_fraction = 1
    simulation_file = "snapshot_0613_101159_simulation_n100_nt5000_nx100_ny100_dt0.0001_dmin0.1_ntsensor20_dmax0.3_nblobs200_radius5_randomTrue.pt"
    simulation_file_path = os.path.join(script_dir, "datasets", "simulation", simulation_file)
    simulation_file = simulation_file.replace(".pt", "")
    
    num_points_x = 2
    num_points_y = 2
    x_coords = (torch.arange(num_points_x) + 0.5) / num_points_x
    y_coords = (torch.arange(num_points_y) + 0.5) / num_points_y
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    sensor_coordinates = torch.stack([xx.flatten(), yy.flatten()], dim=1).transpose(0, 1)

    print("sensor coordinates shape:", sensor_coordinates.shape)
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
    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) > 1:
        train_dataset, test_dataset = split(dataset, 0.8)
    else:
        train_dataset = dataset

    # --- Configuration ---
    log_tensorboard = True
    save_model = False
    TRAIN_IN_STAGES = False
    epochs = 200
    EMBED_DIM = 16  # dimensionality of temporal embedding produced by the LSTM
    trunk_depth = 16
    branch_depth = 16
    trunk_width = 48
    branch_width = 48
    batch_size = 32
    weight_decay = 0
    lstm = True

    # --- Model & HParams ---
    num_sensors = sensor_coordinates.shape[1]
    new_u_shape = TensorShape(dim=1, size=([EMBED_DIM]))
    operator_shapes = OperatorShapes(x=dataset.shapes.x, u=new_u_shape, y=dataset.shapes.y, v=dataset.shapes.v)
    operator = DeepCatOperator(shapes=operator_shapes, device=device, trunk_depth=trunk_depth, branch_depth=branch_depth, trunk_width=trunk_width, branch_width=branch_width)
    operator = operator.to(device)
    lstm_network = LSTM(input_size=num_sensors, embed_dim=EMBED_DIM).to(device) if lstm else None

    total_params = sum(p.numel() for p in operator.parameters())
    if lstm and lstm_network:
        total_params += sum(p.numel() for p in lstm_network.parameters())
    print("total param: ", total_params)

    hparams = {
        'trunk_depth': trunk_depth, 'branch_depth': branch_depth, 'trunk_width': trunk_width, 'branch_width': branch_width,
        'epochs': epochs, 'dataset_size': len(dataset), 'model_type': type(operator).__name__, 'total_params': total_params,
        'dataset_name': data_file_name, 'weight_decay': weight_decay, 'scheduler': 'LinearLRScheduler',
        'batch_size': batch_size, 'lstm': lstm, 'embedding_dim': EMBED_DIM, 'training_mode': 'staged' if TRAIN_IN_STAGES else 'joint'
    }

    # --- DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    mse_loss = torch.nn.MSELoss()
    
    # --- Training with Exception Handling ---
    log_dir = f"runs/{timestamp}"
    writer = None  # Initialize to None

    try:
        if log_tensorboard:
            writer = SummaryWriter(log_dir=log_dir)
            hparam_text = "\n".join(f"{key}: {value}" for key, value in hparams.items())
            writer.add_text("Hyperparameters", hparam_text)
            print(f"TensorBoard logs will be saved to: {log_dir}")

        if TRAIN_IN_STAGES:
            if not lstm:
                raise ValueError("Staged training requires the 'lstm' flag to be True.")
            train_stage_1(operator, train_loader, val_loader, epochs, weight_decay, device, log_dir, log_tensorboard, mse_loss)
            train_stage_2(operator, lstm_network, train_loader, val_loader, epochs, weight_decay, device, log_dir, log_tensorboard, mse_loss)
            train_stage_3(operator, lstm_network, train_loader, val_loader, epochs, weight_decay, device, log_dir, log_tensorboard, mse_loss)
        
        else: # Joint training
            train_jointly(
                operator=operator, lstm_network=lstm_network, train_loader=train_loader, val_loader=val_loader,
                epochs=epochs, weight_decay=weight_decay, device=device, base_log_dir=log_dir,
                log_tensorboard=log_tensorboard, loss_fn=mse_loss, use_lstm_flag=lstm
            )

    except (KeyboardInterrupt, Exception) as e:
        print("\n--- Training Interrupted ---")
        if writer:
            writer.close()
        
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"Removed incomplete logs at: {log_dir}")
        
        if isinstance(e, KeyboardInterrupt):
            print("Training stopped by user.")
            sys.exit(1)
        else:
            print(f"An error occurred during training: {e}")
            raise

    else:
        # This block runs only if the try block completes without exceptions.
        if writer:
            writer.close()
        print("\n--- Training Completed Successfully ---")
        print(f"Logs saved at: {log_dir}")

        # VISUALISATION AFTER SUCCESSFUL TRAINING
        print("\n--- Generating Visualizations ---")
        visualize_predictions(operator, lstm_network, train_dataset, num_samples=5, mode="train", device=device, log_dir=log_dir)
        visualize_predictions(operator, lstm_network, test_dataset, num_samples=5, mode="test", device=device, log_dir=log_dir)


if __name__ == "__main__":
    main()
