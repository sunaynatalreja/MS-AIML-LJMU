import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import joblib
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import GRUClassifier


def plot_graphs(epochs,train_loss,val_loss,train_acc,val_acc):
    plt.figure()
    plt.plot(epochs, train_loss, marker='o', color='blue')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('train_loss.png')
    plt.close()


    plt.figure()
    plt.plot(epochs, val_loss, marker='o', color='orange')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('val_loss.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, marker='o', color='green')
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('train_accuracy.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, val_acc, marker='o', color='red')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('val_accuracy.png')
    plt.close()

def plot_confusion_matrix(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

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
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    all_preds = []
    all_labels = []
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

        val_acc,val_loss,actual_labels,preds = evaluate(model, val_loader, device)
        all_preds.extend(preds)
        all_labels.extend(actual_labels)
        train_losses.append(total_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model with val acc: {best_acc:.2f}%")

    epochs_range = list(range(1, epochs + 1))
    plot_graphs(epochs_range, train_losses, val_losses, train_accuracies, val_accuracies)
    plot_confusion_matrix(all_labels,all_preds)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    loss=0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = 100 * correct / total
    print(f"Validation Acc: {acc:.2f}%")
    return acc,loss,all_labels, all_preds

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
