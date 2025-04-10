import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import requests
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM model for stock price prediction"""
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    """Transformer model for stock price prediction"""
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_size=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding is handled implicitly as we're using linear embedding
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        # Create mask for padding if needed (not used in this simplified version)
        # mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # Use the output of the last position
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # Project to output size
        output = self.output_layer(x)  # [batch_size, output_size]
        
        return output

class StockPredictor:
    """Stock price prediction using either LSTM or Transformer models"""
    
    def __init__(self, model_type="lstm", sequence_length=30, prediction_days=30, device=None):
        """
        Initialize the stock predictor
        
        Args:
            model_type: either "lstm" or "transformer"
            sequence_length: number of days to use for prediction
            prediction_days: number of days to predict into the future
            device: torch device, if None will use CUDA if available
        """
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Initialize model based on type
        if self.model_type == "lstm":
            self.model = LSTMModel(input_size=1, hidden_layer_size=64, num_layers=2, output_size=1)
        elif self.model_type == "transformer":
            self.model = TransformerModel(input_size=1, d_model=64, nhead=4, num_layers=2, output_size=1)
        else:
            raise ValueError(f"Model type '{model_type}' not supported. Use 'lstm' or 'transformer'.")
            
        self.model.to(self.device)
        
    def _get_stock_data(self, symbol, from_date=None, to_date=None):
        """
        Get historical stock data from an API (placeholder function)
        
        In a real application, this would fetch data from an API like Alpha Vantage, Yahoo Finance, etc.
        For demo purposes, we'll generate some dummy data
        """
        # Generate random stock data for demonstration
        if from_date is None:
            from_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
        if to_date is None:
            to_date = datetime.now()
            
        # Number of days between dates
        days = (to_date - from_date).days
        
        # Generate dates
        date_range = [from_date + timedelta(days=i) for i in range(days)]
        
        # Generate prices with some random but realistic movement
        price = 100.0  # Starting price
        prices = [price]
        
        for _ in range(days-1):
            # Random daily return between -2% and +2% with slight upward bias
            daily_return = np.random.normal(0.0005, 0.015)
            price *= (1 + daily_return)
            prices.append(price)
            
        # Create DataFrame
        stock_data = pd.DataFrame({
            'Date': date_range,
            'Close': prices
        })
        
        return stock_data
    
    def prepare_data(self, stock_data, train_size=0.8):
        """Prepare data for training"""
        # Extract close prices and convert to numpy array
        prices = stock_data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences for training
        X, y = [], []
        
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i+self.sequence_length])
            y.append(scaled_data[i+self.sequence_length])
            
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * train_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        # Create DataLoaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        return train_loader, test_loader
    
    def train(self, train_loader, epochs=50, learning_rate=0.001):
        """Train the model"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data"""
        self.model.eval()
        criterion = nn.MSELoss()
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += criterion(output, target).item()
                
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.6f}")
        
        return test_loss
    
    def predict(self, symbol, days=None):
        """Make predictions for the given stock symbol"""
        if days is None:
            days = self.prediction_days
            
        # Get historical data
        historical_data = self._get_stock_data(symbol)
        
        # Prepare the latest sequence for prediction
        latest_prices = historical_data['Close'].values[-self.sequence_length:].reshape(-1, 1)
        scaled_data = self.scaler.transform(latest_prices)
        
        # Initialize predictions array
        predictions = []
        current_batch = scaled_data.copy()
        
        # Make predictions for the specified number of days
        self.model.eval()
        with torch.no_grad():
            for _ in range(days):
                # Prepare the input tensor
                current_batch_tensor = torch.tensor(current_batch.reshape(1, self.sequence_length, 1), 
                                                  dtype=torch.float32).to(self.device)
                
                # Make prediction
                pred = self.model(current_batch_tensor)
                
                # Add prediction to the list
                predictions.append(pred.cpu().numpy()[0][0])
                
                # Update the batch for the next prediction
                current_batch = np.append(current_batch[1:], [[pred.cpu().numpy()[0][0]]], axis=0)
                
        # Inverse transform the predictions to get actual prices
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.scaler.inverse_transform(predictions_array)
        
        # Generate dates for predictions
        last_date = historical_data['Date'].iloc[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # Create DataFrame with predictions
        prediction_df = pd.DataFrame({
            'Date': prediction_dates,
            'Price': predictions_rescaled.flatten(),
            # Add a confidence score (this is a placeholder - real models would have proper confidence intervals)
            'Confidence': np.linspace(0.95, 0.70, days)
        })
        
        return prediction_df
    
    def save_model(self, path):
        """Save the model to the specified path"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'scaler': self.scaler
        }, path)
        
    def load_model(self, path):
        """Load the model from the specified path"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update model parameters
        self.model_type = checkpoint['model_type']
        self.sequence_length = checkpoint['sequence_length']
        self.prediction_days = checkpoint['prediction_days']
        self.scaler = checkpoint['scaler']
        
        # Reinitialize the model based on loaded type
        if self.model_type == "lstm":
            self.model = LSTMModel(input_size=1, hidden_layer_size=64, num_layers=2, output_size=1)
        elif self.model_type == "transformer":
            self.model = TransformerModel(input_size=1, d_model=64, nhead=4, num_layers=2, output_size=1)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval() 