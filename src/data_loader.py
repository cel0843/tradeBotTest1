"""
Data loader module for downloading and caching stock market data.
Uses yfinance as the primary data source.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def download_ohlcv_yfinance(
    symbol: str,
    start: str,
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format, None for today
    
    Returns:
        DataFrame with columns: open, high, low, close, adj_close, volume
        Index: Date (timezone-naive)
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=False)
        
        if df.empty:
            raise ValueError(f"No data retrieved for {symbol}")
        
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Rename columns to standard format
        column_mapping = {
            'adj close': 'adj_close'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Select and order columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if 'adj_close' in df.columns:
            required_cols.insert(4, 'adj_close')
        
        df = df[required_cols].copy()
        
        # Ensure timezone-naive datetime index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Remove any duplicate dates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Forward fill missing values
        df.fillna(method='ffill', inplace=True)
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"Failed to download data for {symbol}: {str(e)}")


def load_or_download(
    symbol: str,
    cache_dir: Path,
    start: str,
    end: Optional[str] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load data from cache or download if not available.
    
    Args:
        symbol: Stock ticker symbol
        cache_dir: Directory to store cached data
        start: Start date for data
        end: End date for data, None for today
        force_download: If True, bypass cache and download fresh data
    
    Returns:
        DataFrame with OHLCV data
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{symbol}.csv"
    
    # Check if cache exists and is recent
    if cache_file.exists() and not force_download:
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            # Check if cached data is up to date
            last_date = df.index[-1]
            today = pd.Timestamp.now().normalize()
            
            # If data is from today or yesterday (accounting for market hours), use cache
            if (today - last_date).days <= 1:
                print(f"Loading {symbol} from cache (last update: {last_date.date()})")
                return df
            else:
                print(f"Cache outdated for {symbol}, downloading fresh data...")
        except Exception as e:
            print(f"Error reading cache for {symbol}: {e}, downloading fresh data...")
    
    # Download fresh data
    print(f"Downloading {symbol} data from {start} to {end or 'today'}...")
    df = download_ohlcv_yfinance(symbol, start, end)
    
    # Save to cache
    try:
        df.to_csv(cache_file)
        print(f"Saved {symbol} data to cache ({len(df)} rows)")
    except Exception as e:
        print(f"Warning: Could not save cache for {symbol}: {e}")
    
    return df


def get_latest_price(symbol: str) -> float:
    """
    Get the latest price for a symbol.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Latest closing price
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return float(data['Close'].iloc[-1])
        else:
            raise ValueError(f"No price data available for {symbol}")
    except Exception as e:
        raise RuntimeError(f"Failed to get latest price for {symbol}: {str(e)}")


def download_multiple_symbols(
    symbols: list,
    start: str,
    end: Optional[str] = None,
    cache_dir: Optional[Path] = None
) -> dict:
    """
    Download data for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        start: Start date
        end: End date, None for today
        cache_dir: Cache directory, None to skip caching
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    data_dict = {}
    
    for symbol in symbols:
        try:
            if cache_dir:
                df = load_or_download(symbol, cache_dir, start, end)
            else:
                df = download_ohlcv_yfinance(symbol, start, end)
            
            data_dict[symbol] = df
            print(f"✓ {symbol}: {len(df)} rows")
        except Exception as e:
            print(f"✗ {symbol}: {str(e)}")
    
    return data_dict


if __name__ == "__main__":
    from config import DATA_DIR, DATA_START, DATA_END
    
    print("=== Data Loader Test ===\n")
    
    # Test single symbol download
    symbol = "AAPL"
    print(f"Testing download for {symbol}...")
    
    df = load_or_download(
        symbol=symbol,
        cache_dir=DATA_DIR,
        start=DATA_START,
        end=DATA_END
    )
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    print(f"\nData info:")
    print(df.info())
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Test latest price
    try:
        latest = get_latest_price(symbol)
        print(f"\nLatest price for {symbol}: ${latest:.2f}")
    except Exception as e:
        print(f"\nCould not get latest price: {e}")
