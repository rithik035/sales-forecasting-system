import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = None
        self.features = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_4', 'rolling_std_4', 'day_of_week', 'month', 'year', 'is_holiday']

    def train(self, train_data):
        X = train_data[self.features]
        y = train_data['Total']
        
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.model.fit(X, y)
        return self.model

    def predict(self, steps, last_known_row=None):
        if self.model is None:
            raise ValueError("Model must be trained before predicting.")
        
        # Simple implementation: use the features from last_known_row to predict multiple steps
        # In a real scenario, we would update lags iteratively.
        # For this case study, we'll assume the features (except lags) are predictable (date parts)
        # and we'll use a simplified recursive update for lags.
        
        predictions = []
        current_features = last_known_row[self.features].copy()
        
        for i in range(steps):
            pred = self.model.predict(current_features.values.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Update lags for next step (simplified)
            # lag_1 becomes the current prediction
            current_features['lag_1'] = pred
            # In a full implementation, we'd update lag_7, lag_30, rolling stats too.
            # But that requires more state. For now, we'll just update lag_1 and keep others.
            
        return np.array(predictions)

    def evaluate(self, test_data):
        X_test = test_data[self.features]
        y_test = test_data['Total']
        predictions = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"rmse": rmse, "mae": mae, "predictions": predictions}
