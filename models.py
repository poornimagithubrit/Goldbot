import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression

def compute_7_day_avg_price(df):
    df['Sale Date'] = pd.to_datetime(df['Sale Date'], errors='coerce')
    recent = df[df['Sale Date'] >= df['Sale Date'].max() - timedelta(days=7)]
    recent = recent[recent['Net Weight'] > 0]
    recent['Price_per_gram'] = recent['Net Sales Value [INR]'] / recent['Net Weight']
    weighted_avg = np.average(recent['Price_per_gram'], weights=recent['Net Weight'])
    return round(weighted_avg, 2)

def forecast_next_month(df):
    df = df[df['Net Weight'] > 0].copy()
    df['Sale Date'] = pd.to_datetime(df['Sale Date'], errors='coerce')
    df['Price_per_gram'] = df['Net Sales Value [INR]'] / df['Net Weight']

    daily_avg = df.groupby('Sale Date')['Price_per_gram'].mean().reset_index()
    daily_avg['Day_Index'] = (daily_avg['Sale Date'] - daily_avg['Sale Date'].min()).dt.days

    model = LinearRegression()
    model.fit(daily_avg[['Day_Index']], daily_avg['Price_per_gram'])

    future = pd.DataFrame({
        'Day_Index': range(daily_avg['Day_Index'].max() + 1, daily_avg['Day_Index'].max() + 31)
    })
    future['Date'] = pd.date_range(daily_avg['Sale Date'].max() + timedelta(days=1), periods=30)
    future['Forecasted Price per gram'] = model.predict(future[['Day_Index']])

    return future[['Date', 'Forecasted Price per gram']]
