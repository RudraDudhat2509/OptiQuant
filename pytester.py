import pandas as pd
import numpy as np
import pytest
from DataPreprocessing import add_features

class TestOptiQuantArchitecture:
    """
    Production unit tests to strictly enforce architectural boundaries 
    and prevent data leakage in the ML pipeline.
    """
    
    @pytest.fixture
    def dummy_market_data(self):
        """Generates 100 days of random walk data for two tickers."""
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", periods=100)
        
        data = []
        for ticker in ['AAPL', 'MSFT']:
            prices = np.cumprod(1 + np.random.normal(0, 0.02, 100)) * 100
            volumes = np.random.randint(1000000, 5000000, 100)
            
            df = pd.DataFrame({
                'Date': dates,
                'Name': ticker,
                'open': prices * np.random.normal(1, 0.005, 100),
                'high': prices * np.random.normal(1.01, 0.005, 100),
                'low': prices * np.random.normal(0.99, 0.005, 100),
                'close': prices,
                'volume': volumes
            })
            data.append(df)
            
        return pd.concat(data, ignore_index=True)

    def test_no_future_data_leakage(self, dummy_market_data):
        """
        CRITICAL TEST: Ensures no feature at time T relies on the Target at time T+5.
        """
        # Run the feature engineering pipeline
        processed_df = add_features(dummy_market_data)
        
        # Isolate a single ticker to check chronological logic
        aapl_df = processed_df[processed_df['Name'] == 'AAPL'].sort_values('Date').reset_index(drop=True)
        
        # 1. Target Validation: Target MUST be NaN for the last 5 days
        last_5_targets = aapl_df['Target'].tail(5)
        assert last_5_targets.isna().all(), "Data Leakage Alert: Target column has values for the last 5 days, indicating an invalid forward shift."

        # 2. Feature Validation: Features MUST NOT be NaN for the last 5 days 
        # (meaning they don't depend on the NaN targets)
        feature_cols = ['Momentum_5d', 'Volatility_20d', 'Beta_20d', 'RSI_14']
        
        for feature in feature_cols:
            # We check the very last day. It should have a valid feature value 
            # if enough historical data exists, even though the Target is NaN.
            last_day_feature = aapl_df.iloc[-1][feature]
            assert not pd.isna(last_day_feature), f"Data Leakage Alert: {feature} is NaN on the last day. It may be improperly coupled to the Target."

if __name__ == "__main__":
    # To run this locally: 
    # pip install pytest
    # pytest test_architecture.py -v
    print("Unit test suite initialized. Run via pytest.")