from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Form
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from openai_agent import get_intent_from_prompt

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ Intent alias mapping
INTENT_ALIASES = {
    "get_7_day_avg": [
        "get_7_day_avg", "get_7_day_avg_per_gram", "7_day_average",
        "7_day_avg_price", "last_7_days_average", "gold_avg_week"
    ],
    "forecast_next_month": [
        "forecast_next_month", "forecast_gold_prices_next_month",
        "predict_next_month", "predict_gold_trend_next_month",
        "gold_price_next_month", "price_forecast"
    ]
}


def resolve_intent(raw_intent: str) -> str:
    """Map incoming intent to canonical action"""
    raw_intent = raw_intent.strip().lower()
    for canonical, variants in INTENT_ALIASES.items():
        if raw_intent in variants:
            return canonical
    return "unrecognized"


def preprocess_sales_data(file):
    """Load and preprocess uploaded CSV file"""
    try:
        df = pd.read_csv(file)

        # Check required columns
        required_cols = ['Sale Date', 'Net Sales Value [INR]', 'Net Weight']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns in CSV")

        df = df[df['Net Weight'] > 0].copy()
        df['Sale Date'] = pd.to_datetime(df['Sale Date'], errors='coerce')
        df.dropna(subset=['Sale Date'], inplace=True)
        df['Price_per_gram'] = df['Net Sales Value [INR]'] / df['Net Weight']

        return df
    except Exception as e:
        raise RuntimeError(f"CSV load/preprocess error: {e}")


@app.post("/query/")
async def query_gold(prompt: str = Form(...), file: UploadFile = File(...)):
    try:
        df = preprocess_sales_data(file.file)
    except Exception as e:
        return {"error": str(e)}

    start = time.time()
    raw_intent = get_intent_from_prompt(prompt)
    intent = resolve_intent(raw_intent)

    if intent == "get_7_day_avg":
        result = compute_7_day_avg_price(df)

    elif intent == "forecast_next_month":
        try:
            result = forecast_next_month(df).to_dict(orient="records")
        except Exception as e:
            result = {"error": str(e)}

    else:
        result = {"message": f"Unrecognized prompt intent: '{raw_intent}'"}

    elapsed = round(time.time() - start, 2)
    return {"intent": raw_intent, "resolved_intent": intent, "result": result, "execution_time": f"{elapsed} seconds"}


def compute_7_day_avg_price(df):
    recent = df[df['Sale Date'] >= df['Sale Date'].max() - timedelta(days=7)]
    recent = recent[recent['Net Weight'] > 0]
    weighted_avg = np.average(recent['Price_per_gram'], weights=recent['Net Weight'])
    return round(weighted_avg, 2)


def forecast_next_month(df):
    daily_avg = df.groupby('Sale Date', as_index=False)['Price_per_gram'].mean()
    daily_avg = daily_avg.dropna().drop_duplicates(subset='Sale Date').sort_values('Sale Date')

    if len(daily_avg) < 10:
        raise ValueError("Not enough historical data to forecast")

    daily_avg['Day_Index'] = (daily_avg['Sale Date'] - daily_avg['Sale Date'].min()).dt.days

    model = LinearRegression()
    model.fit(daily_avg[['Day_Index']], daily_avg['Price_per_gram'])

    max_day = daily_avg['Day_Index'].max()
    future_days = pd.DataFrame({'Day_Index': range(max_day + 1, max_day + 31)})
    future_days['Date'] = pd.date_range(start=daily_avg['Sale Date'].max() + timedelta(days=1), periods=30)

    future_days['Forecasted Price per gram'] = model.predict(future_days[['Day_Index']]).round(2)

    return future_days[['Date', 'Forecasted Price per gram']]
