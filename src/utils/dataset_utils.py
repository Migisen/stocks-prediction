import pathlib
import pandas as pd

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values

import numpy as np

class DatasetError(Exception):
    ...

class DatasetUtils:
    def __init__(self, date_col_name: str = '<DATE>', time_col_name: str = '<TIME>', vol_col_name: str = '<VOL>'):
        self.date_col_name, self.time_col_name = date_col_name, time_col_name
        self.vol_col_name = vol_col_name
    
    def get_hourly_df(self, dataset_path: pathlib.Path, sep: str = ',') -> pd.DataFrame:
        hourly_df = DatasetUtils.load_raw_data(dataset_path, sep)
        hourly_df = DatasetUtils.create_datetime(hourly_df, self.date_col_name, self.time_col_name)
        return hourly_df
            
    def get_daily_df(self, dataset_path: pathlib.Path, sep: str = ',') -> pd.DataFrame:
        hourly_df = self.get_hourly_df(dataset_path, sep)
        max_time_df = hourly_df[[self.date_col_name, self.time_col_name]].groupby(self.date_col_name).max()
        max_time_df.reset_index(drop=False, inplace=True)
        daily_df = max_time_df.merge(hourly_df, on=[self.date_col_name, self.time_col_name])
        
        # Adding volume
        daily_volume = DatasetUtils.get_daily_volume(hourly_df, self.date_col_name, self.vol_col_name)
        daily_df.drop(self.vol_col_name, axis=1, inplace=True)
        daily_df = daily_df.merge(daily_volume, on=self.date_col_name)
        daily_df = DatasetUtils.create_datetime(daily_df, self.date_col_name, self.time_col_name, replace_zeros=False)
        daily_df.index = daily_df.index.normalize()
        return daily_df
    
    def create_daily_dart_timeseries(self, dataset_path: pathlib.Path, fill_missing: bool = True, 
                                     remove_structural: bool = False, structural_date: str = '2022-03-1',
                                     crush_start_date: str = '2022-02-18', target_col_name: str = '<CLOSE>'):
        dart_timeseries = TimeSeries.from_dataframe(self.get_daily_df(dataset_path)[[target_col_name]], fill_missing_dates=True, freq='D')
        if fill_missing:
            dart_timeseries = fill_missing_values(dart_timeseries)
        if remove_structural:
            first_half = dart_timeseries[:pd.to_datetime(structural_date)]
            second_half = dart_timeseries[pd.to_datetime(structural_date):]
            first_half = first_half - (np.mean(first_half.values()) - np.mean(second_half.values()))
            dart_timeseries = first_half[:pd.to_datetime(crush_start_date)].concatenate(second_half, ignore_time_axis=True)
            dart_timeseries = fill_missing_values(dart_timeseries)
        return dart_timeseries
    
    @staticmethod
    def get_train_val_test_split(timeseries: TimeSeries, test_size: float, val_size: float):
        train_val_series, test_series = timeseries.split_before(1 - test_size)
        train_series, val_series = train_val_series.split_before(1 - val_size)
        return train_series, val_series, test_series


    
    @staticmethod
    def get_daily_volume(daily_df: pd.DataFrame, date_col_name: str, vol_col_name: str) -> pd.DataFrame:
        return daily_df[[date_col_name, vol_col_name]].groupby(date_col_name).sum()

    @staticmethod
    def create_datetime(df: pd.DataFrame, date_col_name: str, time_col_name: str, replace_zeros: bool = True, dt_frmt: str = '%Y%m%d%H%M%S') -> pd.DataFrame:
        if replace_zeros:
            df[time_col_name] = df[time_col_name].apply(lambda x: 100000 if x == 0 else x)
        df['<DATETIME>'] = pd.to_datetime(df[date_col_name].astype(str) + df[time_col_name].astype(str), format=dt_frmt)
        # TODO: mb add sort by datetime
        df.set_index('<DATETIME>', inplace=True)
        return df

    @staticmethod
    def load_raw_data(dataset_path: pathlib.Path, sep: str = ','):
        try:
            raw_df = pd.read_csv(dataset_path, sep=sep)
        except Exception as e:
            print('Failed to load dataset')
            raise DatasetError(str(e))
        return raw_df