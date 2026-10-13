# GenAI Acknowledgment: Used Gemini for script architecture, PyTorch training loop design, and automated README generation.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    # Part 1: Data Loading and Exploration
    print("=== DATASET DESCRIPTION ===")
    california = fetch_california_housing()
    print(california.DESCR)

    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MedHouseVal'] = california.target

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== SUMMARY STATISTICS ===")
    print(df.describe())

    # Part 2: Data Preprocessing
    X = df.drop('MedHouseVal', axis=1).values
    y = df['MedHouseVal'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Part 3: Model Building and Training
    # Model 1: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Model 2: Neural Network with PyTorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()
            self.hidden = nn.Linear(input_dim, 32)
            self.relu = nn.ReLU()
            self.output = nn.Linear(32, 1)

        def forward(self, x):
            x = self.relu(self.hidden(x))
            x = self.output(x)
            return x

    mlp_model = MLP(X_train_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

    print("\n=== PYTORCH MLP TRAINING ===")
    epochs = 100
    losses = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = mlp_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    # Part 4: Model Evaluation
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_rmse = lr_mse ** 0.5

    mlp_model.eval()
    with torch.no_grad():
        mlp_predictions = mlp_model(X_test_tensor)
        mlp_mse = criterion(mlp_predictions, y_test_tensor).item()
        mlp_rmse = mlp_mse ** 0.5

    print("\n=== EVALUATION METRICS ===")
    print(f"Linear Regression -> MSE: {lr_mse:.4f}, RMSE: {lr_rmse:.4f}")
    print(f"PyTorch MLP       -> MSE: {mlp_mse:.4f}, RMSE: {mlp_rmse:.4f}")

    # Plot Loss
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(range(1, epochs + 1), losses, color='blue')
    plt.title('PyTorch MLP Training Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'loss_plot.png')
    plt.savefig(plot_path)
    print(f"\nGenerated: {plot_path}")

if __name__ == "__main__":
    main()