"""Preparation of dataframe for deep learning."""


from typing import Tuple

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo


def fetch_data() -> pd.DataFrame:
    """Fetch dataframe from UCI repository."""
    try:
        data = pd.read_csv('../data/household_power_consumption.txt', sep=';',
                           low_memory=False)
    except FileNotFoundError:
        individual_household_electric_power_consumption = fetch_ucirepo(id=235)
        data = individual_household_electric_power_consumption.data.features
    return data


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, list, str]:
    """Preprocess dataframe for timeseries analysis."""
    # Format to float
    feats = ['Global_active_power', 'Global_reactive_power', 'Voltage',
             'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
             'Sub_metering_3']
    data[data == '?'] = float('nan')
    for f in feats:
        data[f] = pd.to_numeric(data[f])
    # Handle missing values
    data.loc[data.isna().any(axis=1), feats] = float('nan')
    # Make useful columns
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data['Datetime'] = pd.to_datetime(
        data['Date'].astype(str) + ' ' + data['Time'])
    data['Hour_of_day'] = data['Datetime'].dt.hour
    data['Day_of_year'] = data['Datetime'].dt.day_of_year
    data['Weekday'] = data['Datetime'].dt.weekday
    data = data.reset_index(drop=True)
    # Hourly aggregation (size reduction)
    data = data.drop(columns=['Datetime', 'Time'])
    data = data.groupby(['Date', 'Hour_of_day', 'Day_of_year', 'Weekday'],
                        as_index=False).mean()
    # Features and target
    data['Global_apparent_power'] = np.sqrt(
        data['Global_active_power']**2 + data['Global_reactive_power']**2)
    feats += ['Hour_of_day', 'Day_of_year', 'Weekday']
    target = 'Global_apparent_power'
    return data, feats, target
