import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# =============== Model Definition ===============

class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((input_dim - 1) * 32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class AutoencoderModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# =============== General Training Function ===============

def train_model(model, X_train, Y_train, epochs=100, patience=10, batch_size=32):
    model.train()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model

# =============== DLModels Class ===============

class DLModels:
    def __init__(self, X_train, Y_train):
        self.input_dim = X_train.shape[1]
        self.models = {
            'DNN': self.create_dnn(X_train, Y_train),
            'MLP': self.create_mlp(X_train, Y_train),
            'CNN': self.create_cnn(X_train, Y_train),
            'Autoencoder': self.create_autoencoder(X_train, Y_train)
        }

    def create_dnn(self, X_train, Y_train):
        model = DNNModel(self.input_dim)
        return train_model(model, X_train, Y_train)

    def create_mlp(self, X_train, Y_train):
        model = MLPModel(self.input_dim)
        return train_model(model, X_train, Y_train)

    def create_cnn(self, X_train, Y_train):
        X_train_cnn = X_train[:, np.newaxis, :]
        model = CNNModel(self.input_dim)
        return train_model(model, X_train_cnn, Y_train)

    def create_autoencoder(self, X_train, Y_train):
        model = AutoencoderModel(self.input_dim)
        return train_model(model, X_train, Y_train)

    def evaluate_models(self, X_test, Y_test):
        metrics = {}
        for name, model in self.models.items():
            X_test_in = X_test[:, np.newaxis, :] if name == 'CNN' else X_test
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_test_in, dtype=torch.float32)
                pred = model(X_tensor).cpu().numpy().flatten()

            r2 = r2_score(Y_test, pred)
            rmse = np.sqrt(mean_squared_error(Y_test, pred))
            mae = mean_squared_error(Y_test, pred)
            pearson_corr, _ = pearsonr(Y_test, pred)
            metrics[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Pearson': pearson_corr
            }
        return metrics

    def select_best_model(self, metrics):
        return max(metrics.items(), key=lambda item: item[1]['R2'])

    def DL_model_comparison_scatter(self, X_train, X_test, Y_train, Y_test):
        predictions = {}
        r2_scores = {}
        for name, model in self.models.items():
            X_test_in = X_test[:, np.newaxis, :] if name == 'CNN' else X_test
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_test_in, dtype=torch.float32)
                pred = model(X_tensor).cpu().numpy().flatten()

            predictions[name] = pred
            r2_scores[name] = r2_score(Y_test, pred)

        plt.figure(figsize=(12, 10))
        sns.set_theme(style='whitegrid')

        for i, (name, prediction) in enumerate(predictions.items(), 1):
            plt.subplot(2, 2, i)
            plt.scatter(Y_test, prediction, alpha=0.7, edgecolors='k', s=40, label='Predictions')
            plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', linewidth=1.5)
            plt.title(f"{name} (R²: {r2_scores[name]:.2f})", fontsize=14, fontweight='bold')
            plt.xlabel("True Values", fontsize=12)
            plt.ylabel("Predicted Values", fontsize=12)
            plt.legend(fontsize=10)

        plt.suptitle("Comparison of DL Model Predictions", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return plt.gcf()

# =============== Main Function ===============

def DL_training_main(data, save_dir):
    X = data.iloc[:, 1:].values
    Y = data.iloc[:, 0].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    model_instance = DLModels(X_train, Y_train)
    metrics = model_instance.evaluate_models(X_test, Y_test)
    best_model_name, best_model_metrics = model_instance.select_best_model(metrics)

    os.makedirs(save_dir, exist_ok=True)

    best_model = model_instance.models[best_model_name]
    model_file_path = os.path.join(save_dir, f"{best_model_name}_best_DL_model.pt")
    torch.save(best_model, model_file_path)

    print(f"Best DL model: {best_model_name}, R2: {best_model_metrics['R2']:.4f}, RMSE: {best_model_metrics['RMSE']:.4f}")
    print(f"Best DL model saved at: {model_file_path}")

    fig = model_instance.DL_model_comparison_scatter(X_train, X_test, Y_train, Y_test)
    fig_path = os.path.join(save_dir, "DL_model_comparison_scatter.png")
    fig.savefig(fig_path, dpi=300)
    print(f"DL model comparison plot saved at: {fig_path}")

    return fig, best_model_name, metrics, model_file_path

# =============== CLI ===============

if __name__ == "__main__":
    file_path = r'best_features.xlsx'
    res_df = pd.read_excel(file_path, sheet_name='Sheet1')
    save_dir = r'D:\\Code_Store\\InversionSoftware\\S2_model_training\\Best_Model'
    DL_comparison_scatter_fig, best_model_name, metrics, model_file_path = DL_training_main(res_df, save_dir)
    print(metrics)
    DL_comparison_scatter_fig.show()
