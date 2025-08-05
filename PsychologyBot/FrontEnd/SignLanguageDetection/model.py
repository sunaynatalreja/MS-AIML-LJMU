import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_size=126, hidden_size=64, num_classes=26):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(hidden_size, 128, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.2)
        self.gru3 = nn.GRU(128, 64, batch_first=True, bidirectional=False)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 256)
        self.drop3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop4 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):  # x: (B, T, V, C)
        B, T, V, C = x.shape
        x = x.view(B, T, V * C)  # (B, 50, 126)
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x, _ = self.gru3(x)  # (B, 50, 64)
        x = x[:, -1, :]      # use last time step (B, 64)
        x = self.fc1(x)
        x = self.drop3(x)
        x = self.fc2(x)
        x = self.drop4(x)
        x = self.fc3(x)
        return self.out(x)