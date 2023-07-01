from typing import Generator, List
import pandas as pd 

import numpy as np

class ForecastUtils:
    def __init__(self):
        ...
    
    @staticmethod
    def generate_rolling_window(df: pd.DataFrame, x_col_names: str | List[str], x_periods: int, y_periods: int, offset: int = 1) -> Generator:
        number_steps = int((len(df) - x_periods - y_periods) / offset + 1)
        print(f'Total steps: {number_steps}')
        for idx in range(number_steps):
            if idx != 0:
                idx += offset - 1
            yield df.iloc[idx:idx + x_periods], df.iloc[idx + x_periods:idx + x_periods + y_periods]


if __name__ == "__main__":
    test_df = pd.DataFrame({'dates': pd.date_range('2012-10-01', periods=9, freq='1D'), 'values': np.random.random(9)})
    print(test_df)
    forecast_utils = ForecastUtils()
    for X_train, y_test in forecast_utils.generate_rolling_window(test_df, 'test', 4, 2, 3):
        print(X_train)
        print(y_test)
        print(5*'-')