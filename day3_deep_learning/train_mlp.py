"""day3_deep_learning/train_mlp.py — PyTorch MLP classifier on engineered tabular features"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import full_pipeline_load, scale_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train(epochs=50, lr=1e-3):
    X, y, _ = full_pipeline_load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    Xtr = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    Xte = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    yte = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr)
        loss = loss_fn(out, ytr)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                acc = ((model(Xte) > 0.5).float() == yte).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f} - test_acc: {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "mlp_model.pt"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "mlp_scaler.pkl"))
    joblib.dump(Xtr.shape[1], os.path.join(MODEL_DIR, "mlp_input_dim.pkl"))
    print("MLP model trained and saved.")
    return model


if __name__ == "__main__":
    train()
