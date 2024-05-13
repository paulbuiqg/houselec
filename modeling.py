"""Neural network model for timeseries forecasting."""


from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Forecaster(nn.Module):
    """Neural network model with LSTM and linear fully connected layers."""

    def __init__(self, input_size: int, hidden_size: int, n_layer: int,
                 output_size: int):
        super().__init__()
        try:
            self.checkpoint = torch.load('checkpoint.pt')
            print('Checkpoint loaded')
        except BaseException:
            self.checkpoint = {
                'train_loss_history': [],
                'val_loss_history': [],
                'epochs': 0,
                'best_val_loss': float('inf'),
                'best_epoch': 0,
                'patience_counter': 0
            }
            print('Checkpoint initialized')
        self.lstm = nn.LSTM(input_size, hidden_size, n_layer, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(hidden_size * n_layer, 256)
        self.batchnorm = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        h = torch.permute(h, (1, 0, 2))  # Batch dimension first
        h = self.flatten(h)
        h = self.linear1(h)
        h = self.batchnorm(h)
        h = self.relu(h)
        h = self.dropout(h)
        y = self.linear2(h)
        return y

    def train_one_epoch(
        self,
        device: str,
        dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[torch.optim.Optimizer, list]:
        """Run one training epoch and update optimizer and loss history."""
        self.train()
        loss_history = []
        pbar = tqdm(dataloader, desc='Training', unit='batch')
        for batch in pbar:
            if batch is not None:
                X, y = batch
            else:
                continue
            X, y = X.to(device), y.to(device)
            y_pred = self(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            loss_history.append(loss.item())
            pbar.set_postfix_str(f'loss={loss.item():>8f}')
            del loss
            optimizer.step()
            optimizer.zero_grad()
        return optimizer, loss_history

    def train_many_epochs(
        self,
        device: str,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        n_epoch: int,
        patience: int
    ):
        """Run several training epochs, save checkpoint after each epoch."""
        self.load_checkpoint(optimizer)
        epoch = self.checkpoint['epochs'] + 1
        for epoch in range(epoch, epoch + n_epoch):
            print(f'-- Epoch {epoch} --')
            if self.early_stopping(patience):
                print('Early stopping!')
                break
            optimizer, loss_history = self.train_one_epoch(
                device, dataloader_train, loss_fn, optimizer)
            val_loss = self.evaluate(device, dataloader_val, loss_fn)
            print(f'Evaluation loss: {val_loss:>8f}')
            self.save_checkpoint(optimizer, epoch, loss_history, val_loss)

    def evaluate(
        self,
        device: str,
        dataloader: DataLoader,
        loss_fn: torch.nn.Module,
    ) -> float:
        """Compute (mean) loss from labeled data and their prediction."""
        self.eval()
        pbar = tqdm(dataloader, desc='Evaluation', unit='batch')
        val_loss = 0
        with torch.no_grad():
            for batch in pbar:
                if batch is not None:
                    X, y = batch
                else:
                    continue
                X, y = X.to(device), y.to(device)
                y_pred = self(X)
                loss = loss_fn(y_pred, y)
                val_loss += loss.item() / len(dataloader)
                pbar.set_postfix_str(f'loss={loss.item():>8f}')
        return val_loss

    def load_checkpoint(self, optimizer):
        """Load existing checkpoint or initialize a checkpoint."""
        try:
            self.load_state_dict(self.checkpoint['model_state_dict'])
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        except BaseException:
            pass

    def save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss_history: list,
        val_loss: float
    ):
        """Write checkpoint file."""
        self.checkpoint['model_state_dict'] = self.state_dict()
        self.checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        self.checkpoint['train_loss_history'] += train_loss_history
        self.checkpoint['val_loss_history'] += [val_loss] * \
            len(train_loss_history)
        self.checkpoint['epochs'] += 1
        if val_loss < self.checkpoint['best_val_loss']:
            self.checkpoint['best_model_state_dict'] = self.state_dict()
            self.checkpoint['best_val_loss'] = val_loss
            self.checkpoint['best_epoch'] = epoch
        torch.save(self.checkpoint, 'checkpoint.pt')

    def early_stopping(self, patience: int) -> bool:
        """Check if patience limit has been reached."""
        if self.checkpoint['epochs'] == 0:
            return False
        val_loss = self.checkpoint['val_loss_history'][-1]
        if self.checkpoint['best_val_loss'] < val_loss:
            self.checkpoint['patience_counter'] += 1
        else:
            self.checkpoint['patience_counter'] = 0
        return patience <= self.checkpoint['patience_counter']

    def predict(self, device: str, dataloader: DataLoader
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply model prediction over a labeled dataset."""
        self.eval()
        y_full = []  # Actual values
        y_pred_full = []  # Predicted values
        pbar = tqdm(dataloader, desc='Prediction', unit='batch')
        with torch.no_grad():
            for batch in pbar:
                if batch is not None:
                    X, y = batch
                else:
                    continue
                X, y = X.to(device), y.to(device)
                y_pred = self(X)
                y_full.append(y)
                y_pred_full.append(y_pred)
        return torch.cat(y_full), torch.cat(y_pred_full)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters of model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
