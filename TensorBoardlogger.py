from torch.utils.tensorboard import SummaryWriter
from continuiti.trainer.callbacks import Callback
from continuiti.trainer.callbacks import Logs

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
        print("[TensorBoardLogger] Logging started.")
        self.writer.add_text("info", "Training started with Continuiti", 0)
        
        # Also log hyperparameters as text for easy reading
        if self.hparams:
            hparam_text = "\n".join([f"{k}: {v}" for k, v in self.hparams.items()])
            self.writer.add_text("hyperparameters", hparam_text)
            
            # Log individual hyperparameters as scalars for better tracking
            for name, value in self.hparams.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"hyperparameters/{name}", value)
                    
            print(f"[TensorBoardLogger] Logged {len(self.hparams)} hyperparameters as text")
            # Note: add_hparams will be called at the end of training when we have metrics

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
        if self.hparams:
            # Convert any non-compatible types to strings for TensorBoard
            hparams_compatible = {}
            for k, v in self.hparams.items():
                if isinstance(v, (str, bool, int, float)):
                    hparams_compatible[k] = v
                else:
                    hparams_compatible[k] = str(v)
                    
            self.writer.add_hparams(hparams_compatible, self.final_metrics)
            print("[TensorBoardLogger] Logged hyperparameters with final metrics")
            
        self.writer.close()
        print("[TensorBoardLogger] Logging complete and closed.")