import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter # Added for TensorBoard
from tqdm import tqdm
import time
from dataset import FLRONetDataset
from embedding import Voronoi
from flronet import FLRONetFNO
from sensors import LHS, SensorGenerator


if __name__ == "__main__":
    # ==== Hardcoded Configuration ==== 
    simulation_file = "./datasets/simulation/0530_190751_simulation_n100_nx100_ny100_dt5e-05_dmin0.1_dmax0.3_nblobs200_radius5.pt"
    num_simulations = 10
    sensor_positions_file = "tensors/sensor_positions/pos.pt"
    init_sensor_timeframes = [0, 5, 10, 15, 20]
    init_fullstate_timeframes = [20]
    resolution = (140, 240)
    batch_size = 8
    epochs = 100
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    save_dir = "checkpoints"
    seed = 0

    print("="*60)
    print("FLRONet Training Configuration")
    print("="*60)
    print(f"Device: {device}")
    print(f"Simulation file: {simulation_file}")
    print(f"Number of simulations: {num_simulations}")
    print(f"Resolution: {resolution}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Save directory: {save_dir}")
    print("="*60)

    # Initialize TensorBoard writer (only once!)
    writer = SummaryWriter(log_dir=os.path.join("runs", "flronet", time.strftime("%m%d-%H%M%S")))

    # Ensure checkpoint directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    print("Setting up sensor positions and embedding...")
    # 1) Build your sensor‐position generator
    sensor_gen = LHS(n_sensors=100, device=device)
    # 2) (Temporarily) generate positions so you can pass them to Voronoi
    sensor_gen.resolution = resolution
    sensor_gen.seed       = seed
    sensor_positions      = sensor_gen()

    # 3) Build your embedding generator (dropout / noise are optional)
    emb_gen = Voronoi(
        resolution=resolution, 
        sensor_positions=sensor_positions, 
        dropout_probabilities=[0.01, 0.02, 0.03],  # or [] 
        noise_level=0.05,                          # or 0
        device=device
    )

    print("Loading dataset...")
    # 4) Finally, the dataset
    dataset = FLRONetDataset(
        simulation_file            = simulation_file,
        num_simulations            = num_simulations,
        sensor_generator           = sensor_gen,
        embedding_generator        = emb_gen,
        init_sensor_timeframes     = init_sensor_timeframes,
        init_fullstate_timeframes  = init_fullstate_timeframes,
        resolution                 = resolution,
        seed                       = seed
    )

    print(f"Dataset loaded with {len(dataset)} samples")

    # Train/validation split
    total = len(dataset)
    n_train = int(0.8 * total)
    n_val = total - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train samples: {n_train}, Validation samples: {n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    print("Creating model...")
    # Instantiate model
    model = FLRONetFNO(
        n_channels=1,
        n_fno_layers=2,
        n_hmodes=4,
        n_wmodes=8,
        embedding_dim=32,
        n_stacked_networks=1,
        device=device
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    print("\nStarting training...")
    print("="*60)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", 
                         leave=False, ncols=100)
        
        for batch_idx, (sensor_tfs, sensor_tensor, full_tfs, full_tensor) in enumerate(train_pbar):
            sensor_tfs = sensor_tfs.to(device)
            sensor_tensor = sensor_tensor.to(device)
            full_tfs = full_tfs.to(device)
            full_tensor = full_tensor.to(device)

            preds = model(sensor_tfs, sensor_tensor, full_tfs)
            loss = F.mse_loss(preds, full_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss * sensor_tensor.size(0)
            
            # Update progress bar with current loss
            train_pbar.set_postfix({'Loss': f'{batch_loss:.6f}'})

        avg_train_loss = train_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train', avg_train_loss, epoch) # Log training loss

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        
        # Validation progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for sensor_tfs, sensor_tensor, full_tfs, full_tensor in val_pbar:
                sensor_tfs = sensor_tfs.to(device)
                sensor_tensor = sensor_tensor.to(device)
                full_tfs = full_tfs.to(device)
                full_tensor = full_tensor.to(device)

                preds = model(sensor_tfs, sensor_tensor, full_tfs)
                loss = F.mse_loss(preds, full_tensor)
                batch_loss = loss.item()
                val_loss += batch_loss * sensor_tensor.size(0)
                
                # Update progress bar with current loss
                val_pbar.set_postfix({'Loss': f'{batch_loss:.6f}'})

        avg_val_loss = val_loss / len(val_loader.dataset)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch) # Log validation loss
        epoch_time = time.time() - epoch_start_time

        # Log to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)

        # Print epoch summary
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")

        # Save best checkpoint
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     # # ckpt_path = os.path.join(save_dir, f"flronet_epoch{epoch}_val{avg_val_loss:.6f}.pth")
        #     # torch.save(model.state_dict(), ckpt_path)
        #     # print(f"✓ New best model saved to {ckpt_path}")
        
        print("-" * 80)

    print("="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60)
    writer.close() # Close the TensorBoard writer

