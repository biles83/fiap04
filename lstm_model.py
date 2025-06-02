import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# Hyperparameters

# input_dim = 1      # Number of features in the input data
# hidden_dim = 32     # Number of hidden units in the LSTM
# num_layers = 2       # Number of LSTM layers
# output_dim = 1      # Number of output units (e.g., regression output)
# num_epochs = 500
# batch_size = 64
# lr = 0.01
# seq_length = 30  # Length of the input sequences

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Coletar os dados para treino

symbol = 'PETR4.SA'
start_date = '2020-01-01'
end_date = '2025-05-30'
df = yf.download(symbol, start_date, end_date)

# Preparar os Dados#
df = df.sort_values('Date')
scaler = StandardScaler()
df['Close'] = scaler.fit_transform(df['Close'])
# df['Close']
seq_length = 30
data = []

for i in range(len(df) - seq_length):
    data.append(df.Close[i:i+seq_length])

data = np.array(data)
train_size = int(0.8 * len(data))

x_train = torch.from_numpy(
    data[:train_size, :-1, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(
    data[:train_size, -1, :]).type(torch.Tensor).to(device)
x_test = torch.from_numpy(
    data[train_size:, :-1, :]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(
    data[train_size:, -1, :]).type(torch.Tensor).to(device)

# Treinar Modelo


class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim, device=device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out


model = PredictionModel(input_dim=1, hidden_dim=32,
                        num_layers=2, output_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500

for i in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train)

    if i % 25 == 0:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

y_test_pred = model(x_test)

y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

# Salvar Modelo
torch.save(model.state_dict(), "modelo.pth")
