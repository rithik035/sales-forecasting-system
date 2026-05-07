import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(BaseModel):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, seq_length=12):
        self.model = LSTMNet(input_size, hidden_size, num_layers, 1)
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data[['Total']].values)
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i+self.seq_length])
            y.append(scaled_data[i+self.seq_length])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    def train(self, train_data, epochs=50, lr=0.001):
        X, y = self.prepare_data(train_data)
        X, y = X.to(self.device), y.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        return self.model

    def predict(self, steps, last_sequence=None):
        self.model.eval()
        predictions = []
        
        # Start with the last known sequence
        current_seq = last_sequence.clone().to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(current_seq)
                predictions.append(pred.item())
                # Update sequence: remove first, add prediction at the end
                new_val = pred.view(1, 1, 1)
                current_seq = torch.cat((current_seq[:, 1:, :], new_val), dim=1)
        
        # Inverse scale
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    def evaluate(self, test_data, train_data_for_last_seq=None):
        # We need the last seq_length points from train_data to start predicting
        combined = pd.concat([train_data_for_last_seq.tail(self.seq_length), test_data])
        scaled_combined = self.scaler.transform(combined[['Total']].values)
        
        last_seq = torch.FloatTensor(scaled_combined[:self.seq_length]).view(1, self.seq_length, 1)
        
        steps = len(test_data)
        predictions = self.predict(steps, last_sequence=last_seq)
        
        rmse = np.sqrt(mean_squared_error(test_data['Total'], predictions))
        mae = mean_absolute_error(test_data['Total'], predictions)
        return {"rmse": rmse, "mae": mae, "predictions": predictions}
