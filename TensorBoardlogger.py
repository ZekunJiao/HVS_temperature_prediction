from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger(Callback):
    def __init__(self, log_dir="runs/continuiti", log_weights=False):
        self.writer = SummaryWriter(log_dir)
        self.log_weights = log_weights
        self.operator = None  # will be set in `on_train_begin`
        super().__init__()

    def on_train_begin(self):
        print("[TensorBoardLogger] Logging started.")
        self.writer.add_text("info", "Training started with Continuiti", 0)

    def __call__(self, logs: Logs):
        self.writer.add_scalar("Loss/train", logs.loss_train, logs.epoch)
        if logs.loss_test is not None:
            self.writer.add_scalar("Loss/test", logs.loss_test, logs.epoch)

        if self.log_weights and self.operator:
            for name, param in self.operator.named_parameters():
                self.writer.add_histogram(name, param, logs.epoch)

    def on_train_end(self):
        self.writer.close()
        print("[TensorBoardLogger] Logging complete and closed.")