import torch
from torch.utils.tensorboard import SummaryWriter
from continuiti.trainer.callbacks import PrintTrainingLoss, Logs
import math

class Trainer:
    def __init__(self, operator, lstm_network, optimizer, scheduler, loss_fn, device, writer, use_lstm):
        self.operator = operator
        self.lstm_network = lstm_network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.writer = writer
        self.use_lstm = use_lstm

    def train(self, train_loader, val_loader, epochs):
        steps = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        print_loss_callback = PrintTrainingLoss(epochs, steps)

        for epoch in range(epochs):
            self.operator.train()
            if self.use_lstm:
                self.lstm_network.train()

            loss_train = 0.0
            logs = Logs(epoch=epoch + 1, step=0, loss_train=None, loss_test=None)

            for x, u, y, v in train_loader:
                self.operator.zero_grad()
                x, u, y, v = x.to(self.device), u.to(self.device), y.to(self.device), v.to(self.device)

                if self.use_lstm:
                    u = self.lstm_network(u)
                    u = u[:, -1, :]
                    u = torch.unsqueeze(u, 1)
                else:
                    u = u[:, -1]

                pred = self.operator(x, u, y)
                loss = self.loss_fn(pred, v)
                loss.backward()
                self.optimizer.step()

                loss_train += loss.item()
                logs.step += 1
                logs.loss_train = loss_train / logs.step
                print_loss_callback.step(logs)

            loss_train /= len(train_loader)
            if self.writer:
                self.writer.add_scalar("Loss/Train", loss_train, epoch)

            if val_loader:
                self.evaluate(val_loader, epoch, logs, print_loss_callback)
            
            self.scheduler(logs)

    def evaluate(self, val_loader, epoch, logs, print_loss_callback):
        self.operator.eval()
        if self.use_lstm:
            self.lstm_network.eval()

        loss_eval = 0.0
        with torch.no_grad():
            for x, u, y, v in val_loader:
                x, u, y, v = x.to(self.device), u.to(self.device), y.to(self.device), v.to(self.device)

                if self.use_lstm:
                    u = self.lstm_network(u)
                    u = u[:, -1, :]
                    u = torch.unsqueeze(u, 1)
                else:
                    u = u[:, -1]
                
                pred = self.operator(x, u, y)
                loss = self.loss_fn(pred, v)
                loss_eval += loss.item()

        loss_eval /= len(val_loader)
        logs.loss_test = loss_eval
        print_loss_callback(logs)

        if self.writer:
            self.writer.add_scalar("Loss/Eval", loss_eval, epoch) 