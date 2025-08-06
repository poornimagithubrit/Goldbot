import pandas as pd

REQUIRED_COLUMNS = [
    "Sale Date",
    "Net Sales Value [INR]",
    "Net Weight"
]

def validate_csv_columns(df: pd.DataFrame) -> bool:
    """Check if required columns exist"""
    return all(col in df.columns for col in REQUIRED_COLUMNS)

def preprocess_sales_data(file) -> pd.DataFrame:
    """
    Load and preprocess the uploaded sales CSV:
    - Validates columns
    - Converts Sale Date to datetime
    - Filters rows where Net Weight <= 0
    - Adds computed Price per gram column
    """
    try:
        df = pd.read_csv(file)
        
        if not validate_csv_columns(df):
            raise ValueError(f"Missing required columns: {REQUIRED_COLUMNS}")
        
        df["Sale Date"] = pd.to_datetime(df["Sale Date"], errors="coerce")
        df = df[df["Net Weight"] > 0].copy()
        df["Price_per_gram"] = df["Net Sales Value [INR]"] / df["Net Weight"]

        return df

    except Exception as e:
        raise RuntimeError(f"CSV Processing Failed: {e}")
