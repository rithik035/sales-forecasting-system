import os
import joblib
import pandas as pd
import numpy as np
from src.data_pipeline import DataPipeline
from src.models.arima_model import ArimaModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel

class ForecastingSystem:
    def __init__(self, data_path, output_dir='output'):
        self.pipeline = DataPipeline(data_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.best_models = {} # state -> (model_type, model_object)

    def run_pipeline(self, states=None):
        raw_df = self.pipeline.load_data()
        if states is None:
            states = raw_df['State'].unique()

        results = []

        for state in states:
            print(f"Processing State: {state}")
            processed_df = self.pipeline.preprocess_state_data(raw_df, state)
            featured_df = self.pipeline.create_features(processed_df)
            train, test = self.pipeline.get_train_test_split(featured_df)
            
            if len(test) == 0:
                print(f"Skipping {state} due to insufficient data.")
                continue

            state_results = {}
            
            # 1. ARIMA
            print(f"  Training ARIMA...")
            arima = ArimaModel()
            arima.train(train)
            arima_eval = arima.evaluate(test)
            state_results['ARIMA'] = (arima_eval['rmse'], arima)

            # 2. Prophet
            print(f"  Training Prophet...")
            prophet = ProphetModel()
            prophet.train(train)
            prophet_eval = prophet.evaluate(test)
            state_results['Prophet'] = (prophet_eval['rmse'], prophet)

            # 3. XGBoost
            print(f"  Training XGBoost...")
            xgb_model = XGBoostModel()
            xgb_model.train(train)
            xgb_eval = xgb_model.evaluate(test)
            state_results['XGBoost'] = (xgb_eval['rmse'], xgb_model)

            # 4. LSTM
            print(f"  Training LSTM...")
            lstm = LSTMModel(seq_length=12)
            lstm.train(train, epochs=20) # Low epochs for demo
            lstm_eval = lstm.evaluate(test, train_data_for_last_seq=train)
            state_results['LSTM'] = (lstm_eval['rmse'], lstm)

            # Select best model for this state
            best_model_type = min(state_results, key=lambda k: state_results[k][0])
            best_rmse, best_model_obj = state_results[best_model_type]
            
            self.best_models[state] = {
                'type': best_model_type,
                'rmse': best_rmse,
                'model': best_model_obj
            }
            
            print(f"  Best model for {state}: {best_model_type} (RMSE: {best_rmse:.2f})")
            results.append({'State': state, 'Best Model': best_model_type, 'RMSE': best_rmse})

            # Save all models
            state_slug = state.lower().replace(' ', '_')
            for m_type, (m_rmse, m_obj) in state_results.items():
                joblib.dump(m_obj, os.path.join(self.output_dir, f"{m_type.lower()}_{state_slug}.joblib"))
            
            # Save the best model explicitly as well
            joblib.dump(best_model_obj, os.path.join(self.output_dir, f"best_model_{state_slug}.joblib"))

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        print("\nPipeline completed. Results saved to output directory.")

if __name__ == "__main__":
    # Use relative paths for portability
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'Forecasting Case- Study.xlsx')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    
    system = ForecastingSystem(DATA_PATH, output_dir=OUTPUT_DIR)
    # Run pipeline for all states found in the dataset
    system.run_pipeline()
