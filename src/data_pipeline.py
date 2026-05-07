import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler

class DataPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.us_holidays = holidays.US()

    def load_data(self):
        df = pd.read_excel(self.file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def preprocess_state_data(self, df, state):
        state_df = df[df['State'] == state].copy()
        state_df = state_df.sort_values('Date')
        
        # Resample to weekly frequency (End of week Sunday)
        # We use mean() in case there are multiple entries, but here it's 1 per date
        state_df = state_df.set_index('Date').resample('W-SUN').mean(numeric_only=True)
        
        # Handle missing values
        state_df['Total'] = state_df['Total'].interpolate(method='linear')
        state_df['Total'] = state_df['Total'].ffill().bfill()
        
        return state_df

    def create_features(self, df):
        df = df.copy()
        
        # Lag features
        df['lag_1'] = df['Total'].shift(1)
        df['lag_7'] = df['Total'].shift(7)
        df['lag_30'] = df['Total'].shift(30)
        
        # Rolling features
        df['rolling_mean_4'] = df['Total'].shift(1).rolling(window=4).mean()
        df['rolling_std_4'] = df['Total'].shift(1).rolling(window=4).std()
        
        # Datetime features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['is_holiday'] = df.index.map(lambda x: 1 if x in self.us_holidays else 0)
        
        return df

    def get_train_test_split(self, df, test_size=8):
        # Drop rows with NaN from lags before splitting
        df_clean = df.dropna()
        if len(df_clean) <= test_size:
             # Fallback if not enough data after lags
             return df_clean, pd.DataFrame()
        
        train = df_clean.iloc[:-test_size]
        test = df_clean.iloc[-test_size:]
        return train, test

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'Forecasting Case- Study.xlsx')
    pipeline = DataPipeline(DATA_PATH)
    raw_df = pipeline.load_data()
    print(f"Loaded {len(raw_df)} rows.")
    
    # Test for one state
    state = 'Alabama'
    processed_df = pipeline.preprocess_state_data(raw_df, state)
    featured_df = pipeline.create_features(processed_df)
    train, test = pipeline.get_train_test_split(featured_df)
    
    print(f"State: {state}")
    print(f"Processed rows: {len(processed_df)}")
    print(f"Featured rows: {len(featured_df)}")
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")
    print(featured_df.head(40)) # Should see some NaNs at the beginning
