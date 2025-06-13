import torch.nn as nn
 
class LSTM(nn.Module):
    def __init__(self, input_size: int, embed_dim: int = 16, hidden_size: int = 50, num_layers: int = 1):
        """Simple LSTM encoder producing a D-dimensional temporal embedding.

        Args:
            input_size (int):  Number of sensor channels (num_sensors).
            embed_dim (int):   Desired embedding dimension D.
            hidden_size (int): Hidden size of the LSTM.
            num_layers (int):  Number of stacked LSTM layers.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True)
        self.linear = nn.Linear(hidden_size, embed_dim)
    
    ''' output shape: (batch_size, sequence_length, num_sensor) '''
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x