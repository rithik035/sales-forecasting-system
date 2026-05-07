


Features
- Data Pipeline**: Handles missing dates, resamples to weekly frequency, and implements robust feature engineering (lags, rolling stats, datetime features, holidays).
- Multi-Model Training**: Compares ARIMA/SARIMA, Facebook Prophet, XGBoost, and LSTM.
- Auto-Selection**: Automatically selects and saves the best model per state based on RMSE.
- REST API**: High-performance FastAPI backend for serving 8-week sales forecasts.

Project Structure
- `src/data_pipeline.py`: Data ingestion and feature engineering.
- `src/models/`: Implementation of forecasting algorithms.
- `src/main.py`: Main entry point for training and model selection.
- `api/app.py`: FastAPI application.
- `output/`: Directory for saved models and comparison metrics.

Setup & Usage

1. Install Dependencies
bash
pip install -r requirements.txt


2. Train Models
To train models for specific states (or all states by default):
bash
set PYTHONPATH=.
python src/main.py


3. Start the API
bash
set PYTHONPATH=.
python api/app.py


4. Get Forecasts
Once the API is running,
curl http://localhost:8000/docs


API Endpoints
- `GET /health`: Check system status.
- `GET /forecast/{state}`: Retrieve 8-week sales forecast for the specified state. 
======  example Texas, Alabama etc

