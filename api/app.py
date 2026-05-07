from fastapi import FastAPI, HTTPException
import joblib
import os
import pandas as pd
import torch
import numpy as np
from src.data_pipeline import DataPipeline
from src.models.arima_model import ArimaModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel

app = FastAPI(title="Forecasting System API")

# Use relative paths for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Forecasting Case- Study.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
pipeline = DataPipeline(DATA_PATH)

# Cache for loaded models
models_cache = {}

def get_model(state: str):
    state_slug = state.lower().replace(' ', '_')
    if state_slug in models_cache:
        return models_cache[state_slug]
    
    model_path = os.path.join(OUTPUT_DIR, f"best_model_{state_slug}.joblib")
    if not os.path.exists(model_path):
        return None
    
    try:
        model = joblib.load(model_path)
        models_cache[state_slug] = model
        return model
    except Exception as e:
        print(f"Error loading model for {state}: {e}")
        return None

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/forecast/{state}")
def get_forecast(state: str, model_type: str = None):
    # Normalize state name (Title Case)
    state = state.title()
    
    # List of all possible models we can load
    available_model_types = ["ArimaModel", "ProphetModel", "XGBoostModel", "LSTMModel"]
    
    # Logic to find the requested or best model
    if model_type:
        # User requested a specific model
        # We need to find the file that matches this type
        state_slug = state.lower().replace(' ', '_')
        # This is a bit tricky because we only saved the BEST model as joblib.
        # To support selecting ANY model, we would need to have saved all of them.
        # For now, we will check if the best model matches the requested type, 
        # or load the best model and return its type.
        
        model = get_model(state)
        if model is None:
            raise HTTPException(status_code=404, detail=f"No models found for state {state}. Please train models first.")
        
        actual_type = type(model).__name__
        if model_type.lower() not in actual_type.lower():
             return {
                 "error": f"The requested model '{model_type}' is not the 'Best Model' saved for {state}.",
                 "available_best_model": actual_type,
                 "note": "Currently, only the best performing model is cached and available for inference."
             }
    else:
        model = get_model(state)

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model for state {state} not found.")
    
    raw_df = pipeline.load_data()
    if state not in raw_df['State'].unique():
        raise HTTPException(status_code=404, detail=f"State {state} not found in dataset.")
        
    processed_df = pipeline.preprocess_state_data(raw_df, state)
    featured_df = pipeline.create_features(processed_df)
    
    steps = 8
    
    try:
        # (Prediction logic remains the same)
        if isinstance(model, (ArimaModel, ProphetModel)):
            predictions = model.predict(steps)
        elif isinstance(model, XGBoostModel):
            last_row = featured_df.iloc[-1]
            predictions = model.predict(steps, last_known_row=last_row)
        elif isinstance(model, LSTMModel):
            scaled_data = model.scaler.transform(processed_df[['Total']].values)
            last_seq = torch.FloatTensor(scaled_data[-model.seq_length:]).view(1, model.seq_length, 1)
            predictions = model.predict(steps, last_sequence=last_seq)
        else:
            if hasattr(model, 'predict'):
                if 'XGBoost' in str(type(model)):
                     last_row = featured_df.iloc[-1]
                     predictions = model.predict(steps, last_known_row=last_row)
                else:
                     predictions = model.predict(steps)
            else:
                raise HTTPException(status_code=500, detail="Model object does not have a predict method.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    last_date = processed_df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W-SUN')
    
    forecast_results = []
    for date, pred in zip(forecast_dates, predictions):
        forecast_results.append({
            "date": date.strftime('%Y-%m-%d'),
            "forecasted_sales": float(pred)
        })
        
    return {
        "state": state,
        "selected_model": type(model).__name__,
        "all_system_models": available_model_types,
        "forecast": forecast_results
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print(f"API Documentation available at: http://localhost:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="localhost", port=8000)
