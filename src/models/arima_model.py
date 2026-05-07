import pmdarima as pm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel

class ArimaModel(BaseModel):
    def __init__(self):
        self.model = None

    def train(self, train_data):
        # pmdarima's auto_arima handles model selection
        self.model = pm.auto_arima(train_data['Total'], 
                                   seasonal=True, m=12, # Reduced from 52
                                   error_action='ignore', 
                                   suppress_warnings=True, 
                                   stepwise=True,
                                   max_p=3, max_q=3)
        return self.model

    def predict(self, steps):
        if self.model is None:
            raise ValueError("Model must be trained before predicting.")
        forecast = self.model.predict(n_periods=steps)
        return forecast

    def evaluate(self, test_data):
        steps = len(test_data)
        predictions = self.predict(steps)
        rmse = np.sqrt(mean_squared_error(test_data['Total'], predictions))
        mae = mean_absolute_error(test_data['Total'], predictions)
        return {"rmse": rmse, "mae": mae, "predictions": predictions}
