from forecasters.generic_forecaster import GenericForecaster

import numpy as np
import pandas as pd

from datetime import time, datetime
from typing import List, Optional, Union

class WhiteNoiseForecaster(GenericForecaster):
    def __init__(self, close_name: str = '<CLOSE>', open_name: str = '<OPEN>', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_name = close_name
        self.open_name = open_name
        self.mean = None
        self.std = None
        self.last_date = None
        self.freq_type = None

    def fit(self, x, freq_type: str, **kwargs):
        assert freq_type in ['daily', 'hourly']
        if freq_type == 'daily':
            target_data = x[[self.close_name]]
        else:
            # TODO: change to use proper columns
            target_data = x[[self.close_name]]
        self.freq_type = freq_type
        self.mean = target_data.mean()
        self.std = target_data.std()
        self.last_date = target_data.index.values[-1]

    def predict(self, future_steps, *args, **kwargs) -> pd.DataFrame:
        assert self.mean is not None
        assert self.std is not None
        assert self.last_date is not None
        assert self.freq_type is not None
        predictions = np.random.normal(self.mean, self.std, size=future_steps)
        return predictions
    
    def insert_predictions(self, predictions: List[int | float], future_steps):
        if self.freq_type == "hourly":
            predictions_df = WhiteNoiseForecaster.generate_predict_hourly_df(pd.to_datetime(self.last_date).strftime("%Y-%m-%d"), future_steps)
        elif self.freq_type == "daily":
            predictions_df = WhiteNoiseForecaster.generate_predict_daily_df(pd.to_datetime(self.last_date).strftime("%Y-%m-%d"), future_steps)
        predictions_df['y_pred'] = predictions
        predictions_df.set_index('<DATETIME>', inplace=True)
        return predictions_df

    
    @staticmethod
    def generate_predict_hourly_df(start_date: str, periods: int, hours_per_date: int = 14, start_hour: int = 10, end_hour: int = 23) -> pd.DataFrame:
        # preparing days
        days = pd.bdate_range(start_date, periods=periods, freq='C', weekmask='Sat Mon Tue Wed Thu Fri')
        pred_hourly_df = pd.DataFrame(np.repeat(days.values, hours_per_date, axis=0))
        
        # preparing hours
        hours = np.array([str(time(hour, 0, 0, 0)) for hour in range(start_hour, end_hour + 1)])
        result_hours = np.tile(hours, len(days))[:len(pred_hourly_df)]

        # combining days and hours
        pred_hourly_df['hours'] = result_hours
        pred_hourly_df.columns = ["days", "hours"]

        pred_hourly_df['<DATETIME>'] = pd.to_datetime(pred_hourly_df["days"].astype(str) + pred_hourly_df["hours"].astype(str), format='%Y-%m-%d%H:%M:%S')
        return pred_hourly_df[['<DATETIME>']][:periods]
    
    @staticmethod
    def generate_predict_daily_df(start_date: str, periods: int) -> pd.DataFrame:
        days = pd.bdate_range(start_date, periods=periods, freq='D')
        pred_daily_df = pd.DataFrame(days.values)
        pred_daily_df.columns = ["days"]
        pred_daily_df['<DATETIME>'] = pd.to_datetime(pred_daily_df["days"].astype(str), format='%Y-%m-%d')
        return pred_daily_df[['<DATETIME>']][:periods]
    

class MeanForecaster(WhiteNoiseForecaster):
    def __init__(self, close_name: str = '<CLOSE>', *args, **kwargs):
        super().__init__(close_name=close_name, *args, **kwargs)

    def predict(self, future_steps, *args, **kwargs) -> pd.DataFrame:
        assert self.mean is not None
        assert self.last_date is not None
        assert self.freq_type is not None
        predictions = np.repeat([self.mean], future_steps)
        # self.insert_predictions(predictions, future_steps)
        return predictions
