from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel

class ProphetModel(BaseModel):
    def __init__(self):
        self.model = None

    def train(self, train_data):
        # Prophet expects columns 'ds' and 'y'
        prophet_df = train_data.reset_index()[['Date', 'Total']]
        prophet_df.columns = ['ds', 'y']
        
        self.model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        self.model.fit(prophet_df)
        return self.model

    def predict(self, steps):
        if self.model is None:
            raise ValueError("Model must be trained before predicting.")
        
        future = self.model.make_future_dataframe(periods=steps, freq='W')
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(steps).values

    def evaluate(self, test_data):
        steps = len(test_data)
        predictions = self.predict(steps)
        rmse = np.sqrt(mean_squared_error(test_data['Total'], predictions))
        mae = mean_absolute_error(test_data['Total'], predictions)
        return {"rmse": rmse, "mae": mae, "predictions": predictions}
