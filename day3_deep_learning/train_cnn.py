"""
day3_deep_learning/train_cnn.py
1D-CNN over engineered feature vector (treated as a sequence) — Day3 CNN deliverable.
Saved as models/cnn_model.pt
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from utils.preprocessing import full_pipeline_load, scale_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN1D(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):          # x: (batch, in_dim)
        x = x.unsqueeze(1)         # (batch, 1, in_dim)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def train(epochs=50, lr=1e-3):
    X, y, _ = full_pipeline_load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    Xtr = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    Xte = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    yte = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    model = CNN1D(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                acc = ((model(Xte) > 0.5).float() == yte).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f} - test_acc: {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "cnn_model.pt"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "cnn_scaler.pkl"))
    joblib.dump(Xtr.shape[1], os.path.join(MODEL_DIR, "cnn_input_dim.pkl"))
    print("CNN model trained and saved (Day3 deliverable: models/cnn_model.pt).")
    return model


if __name__ == "__main__":
    train()
