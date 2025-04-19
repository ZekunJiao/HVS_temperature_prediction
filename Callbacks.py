from torch.utils.tensorboard import SummaryWriter
from continuiti.trainer.callbacks import Callback
from continuiti.trainer.callbacks import Logs
import torch
import os


class TensorBoardLogger(Callback):
    def __init__(self, log_dir="runs/continuiti", log_weights=False, hparams=None):
        self.writer = SummaryWriter(log_dir)
        self.log_weights = log_weights
        self.operator = None  # will be set in `on_train_begin`
        self.hparams = hparams or {}
        self.log_dir = log_dir
        self.final_metrics = {}  # Store final metrics for hparams
        super().__init__()

    def on_train_begin(self):        
        # Also log hyperparameters as text for easy reading
        if self.hparams:
            hparam_text = "\n".join([f"{k}: {v}" for k, v in self.hparams.items()])
            self.writer.add_text("hyperparameters", hparam_text)
            

    def __call__(self, logs: Logs):
        self.writer.add_scalar("Loss/train", logs.loss_train, logs.epoch)
        if logs.loss_test is not None:
            self.writer.add_scalar("Loss/test", logs.loss_test, logs.epoch)
            
        # Store the latest metrics for hparams
        self.final_metrics = {
            'hparam/train_loss': logs.loss_train
        }
        if logs.loss_test is not None:
            self.final_metrics['hparam/test_loss'] = logs.loss_test

        if self.log_weights and self.operator:
            for name, param in self.operator.named_parameters():
                self.writer.add_histogram(name, param, logs.epoch)

    def on_train_end(self):
        # Now that we have final metrics, log hyperparameters with metrics
 
        self.writer.add_scalar("final/loss_train", self.final_metrics["hparam/train_loss"])
        self.writer.add_scalar("final/loss_test", self.final_metrics["hparam/test_loss"])
     
        self.writer.close()
        print("[TensorBoardLogger] Logging complete and closed.")


class ModelCheckpointCallback(Callback):
    """
    Callback to save model and optimizer states at specified epoch intervals.

    Args:
        operator (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        save_dir (str): Directory where checkpoints will be saved.
        save_interval (int): Save a checkpoint every `save_interval` epochs.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional):
            Learning rate scheduler to save. Defaults to None.
    """
    def __init__(self, operator, optimizer, save_dir, save_interval=100, scheduler=None):
        self.operator = operator
        self.optimizer = optimizer # Store the optimizer
        self.scheduler = scheduler # Store the scheduler (optional)
        self.save_dir = save_dir
        self.save_interval = save_interval
        os.makedirs(save_dir, exist_ok=True)
        print(f"Checkpoint callback initialized. Saving checkpoints to: {self.save_dir}")
        print(f"Save interval: {self.save_interval} epochs")


    def __call__(self, logs: Logs):
        """
        Saves a checkpoint at the end of an epoch if the interval condition is met.

        Args:
            logs (dict): Dictionary containing training metrics for the epoch.
                         Expected to have an 'epoch' key (0-based or 1-based).
                         Note: Ensure logs.epoch aligns with your training loop's epoch count.
                         If your loop is 0-indexed, epoch 0 % interval == 0.
                         If your loop is 1-indexed, use (logs.epoch - 1) or adjust logic.
        """

        # Check if it's time to save (adjust if epoch is 1-based and you want to save AFTER epoch `save_interval`)
        # Example: If save_interval=100 and epoch is 1-based, save after epoch 100, 200...
        # Using (current_epoch + 1) assumes logs.epoch is 0-based. Adjust if needed.
        if (logs.epoch + 1) % self.save_interval == 0:
            checkpoint_path = os.path.join(
                self.save_dir,
                f"ep{logs.epoch + 1}.pt" # Save using 1-based epoch number in filename
            )

            checkpoint = {
                'epoch': logs.epoch + 1, 
                'model_state_dict': self.operator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }

            # Save scheduler state if it exists
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"\nCheckpoint saved to {checkpoint_path} (Epoch {logs.epoch})")
            except Exception as e:
                print(f"\nError saving checkpoint at epoch {logs.epoch}: {e}")