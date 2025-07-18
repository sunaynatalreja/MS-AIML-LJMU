import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import joblib
import yaml

class GRUClassifier(nn.Module):
    def __init__(self, input_size=126, hidden_size=64, num_classes=151):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.3)
        self.gru2 = nn.GRU(hidden_size, 128, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.3)
        self.gru3 = nn.GRU(128, 64, batch_first=True, bidirectional=False)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 256)
        self.drop3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop4 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):  
        B, T, V, C = x.shape
        x = x.view(B, T, V * C)  
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x, _ = self.gru3(x) 
        x = x[:, -1, :]      
        x = self.fc1(x)
        x = self.drop3(x)
        x = self.fc2(x)
        x = self.drop4(x)
        x = self.fc3(x)
        return self.out(x)


class LandmarkDataset(Dataset):
    def __init__(self, feature_dir, label_map):
        self.samples = []
        self.labels = []
        for file in os.listdir(feature_dir):
            if file.endswith(".npy"):
                label = file.split("_")[0]
                if label in label_map:
                    self.samples.append(os.path.join(feature_dir, file))
                    self.labels.append(label_map[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = np.load(self.samples[idx])  # (T, V, C)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def train(model, train_loader, val_loader, epochs=50, lr=1e-3, save_path="best_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}%")

        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model with val acc: {best_acc:.2f}%")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    print(f"Validation Acc: {acc:.2f}%")
    return acc

# ========== Main ==========
if __name__ == "__main__":
    cwd = os.getcwd()
    config_path = os.path.join(cwd,"PsychologyBot" ,"Config", "config.yaml")
    config = yaml.safe_load(open(config_path))
    model_dir_config=cwd+config['CommonConfig']['ModelDir']
    processed_data_dir=cwd+config['CommonConfig']['ProcessedData']
    DATA_DIR = processed_data_dir+"features/"
    MODEL_DIR=model_dir_config
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3

    class_names = sorted(list(set(f.split("_")[0] for f in os.listdir(DATA_DIR) if f.endswith(".npy"))))
    label_map = {cls: i for i, cls in enumerate(class_names)}
    joblib.dump(label_map, MODEL_DIR+"label_encoding.pkl")

    dataset = LandmarkDataset(DATA_DIR, label_map)
    indices = torch.randperm(len(dataset))
    train_len = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, indices[:train_len])
    val_set = torch.utils.data.Subset(dataset, indices[train_len:])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = GRUClassifier(input_size=126, num_classes=len(label_map))
    train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR,save_path=MODEL_DIR+"best_model.pth")