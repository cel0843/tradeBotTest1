"""
기술적 지표를 생성하고 ML 모델을 위한 데이터를 준비하는 피처 엔지니어링 모듈입니다.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 데이터에 기본적인 기술적 지표와 피처를 추가합니다.
    
    Args:
        df: OHLCV 데이터가 포함된 DataFrame ('close', 'volume' 컬럼 필수)
    
    Returns:
        추가된 피처 컬럼이 포함된 DataFrame
    """
    df = df.copy()
    
    # 일일 수익률
    df['return'] = df['close'].pct_change()
    
    # 이동평균
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    
    # 이동평균 비율
    df['ma5_ratio'] = df['close'] / df['ma5']
    df['ma20_ratio'] = df['close'] / df['ma20']
    df['ma_cross'] = df['ma5'] / df['ma20']
    
    # 변동성 (20일 수익률 표준편차)
    df['volatility'] = df['return'].rolling(window=20).std()
    
    # RSI (상대 강도 지수)
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # 거래량 피처
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # 가격 모멘텀
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # 볼린저 밴드
    bb_period = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std_val = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD (이동평균 수렴 확산)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # ATR (평균 진폭 범위) - 변동성 지표
    df['atr'] = calculate_atr(df, period=14)
    
    # 고가-저가 범위
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    
    # 고점/저점 대비 거리
    df['high_20'] = df['high'].rolling(window=20).max()
    df['low_20'] = df['low'].rolling(window=20).min()
    df['dist_from_high'] = (df['high_20'] - df['close']) / df['close']
    df['dist_from_low'] = (df['close'] - df['low_20']) / df['close']
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI(상대 강도 지수)를 계산합니다.
    
    Args:
        prices: 가격 Series
        period: RSI 기간 (기본값 14)
    
    Returns:
        RSI 값 Series (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 0으로 나누는 경우 처리 (손실이 0일 때)
    rsi = rsi.fillna(100)
    
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR(평균 진폭 범위)을 계산합니다.
    
    Args:
        df: OHLCV DataFrame
        period: ATR 기간
    
    Returns:
        ATR 값 Series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def make_tabular_dataset(
    df: pd.DataFrame,
    target_col: str = 'close',
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    XGBoost와 같은 모델을 위한 테이블형 데이터셋(X, y)을 생성합니다.
    
    Args:
        df: 피처가 포함된 DataFrame
        target_col: 목표 변수 컬럼 (보통 'close')
        horizon: 예측 지평 (1일 후 예측 등)
    
    Returns:
        X (피처), y (라벨), dates (날짜 인덱스), feature_cols (피처 이름 리스트)
    """
    df = df.copy()
    
    # 라벨 생성: (미래 가격 > 현재 가격) = 1, 아니면 0
    # shift(-horizon)은 미래 데이터를 현재 행으로 가져옴
    future_price = df[target_col].shift(-horizon)
    df['target'] = (future_price > df[target_col]).astype(int)
    
    # NaN 제거 (이동평균 등으로 인한 초기 NaN 및 shift로 인한 마지막 NaN)
    df = df.dropna()
    
    if df.empty:
        raise ValueError("NaN 제거 후 데이터가 비어 있습니다. 더 많은 데이터가 필요합니다.")
    
    # 피처 컬럼 선택 (숫자형 컬럼만, 타겟 및 미래 정보 제외)
    exclude_cols = ['target', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols].values
    y = df['target'].values
    dates = df.index
    
    return X, y, dates, feature_cols


def make_supervised_dataset(
    df: pd.DataFrame,
    target_col: str = 'close',
    window_size: int = 20,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    LSTM과 같은 시퀀스 모델을 위한 윈도우 데이터셋을 생성합니다.
    
    Args:
        df: 피처가 포함된 DataFrame
        target_col: 목표 변수 컬럼
        window_size: 입력 시퀀스 길이 (과거 N일)
        horizon: 예측 지평
    
    Returns:
        X (3D 배열: 샘플, 시간, 피처), y (1D 배열), dates (타겟 날짜)
    """
    # 테이블형 데이터셋 생성 로직 재사용하여 기본 정제
    # 여기서는 X, y를 직접 만들지 않고 정제된 df만 사용
    df_clean = df.copy()
    
    # 라벨 생성
    future_price = df_clean[target_col].shift(-horizon)
    df_clean['target'] = (future_price > df_clean[target_col]).astype(int)
    
    # NaN 제거
    df_clean = df_clean.dropna()
    
    if len(df_clean) < window_size:
        raise ValueError(f"데이터 길이가 윈도우 크기({window_size})보다 작습니다.")
    
    # 피처 컬럼 선택
    exclude_cols = ['target', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_clean[c])]
    
    data = df_clean[feature_cols].values
    targets = df_clean['target'].values
    dates = df_clean.index
    
    X_list = []
    y_list = []
    date_list = []
    
    # 슬라이딩 윈도우
    for i in range(len(data) - window_size):
        X_window = data[i : i + window_size]
        y_target = targets[i + window_size - 1] # 윈도우의 마지막 날에 대한 타겟 (다음날 예측)
        target_date = dates[i + window_size - 1]
        
        X_list.append(X_window)
        y_list.append(y_target)
        date_list.append(target_date)
    
    X = np.array(X_list)
    y = np.array(y_list)
    dates = pd.DatetimeIndex(date_list)
    
    return X, y, dates


if __name__ == "__main__":
    # 테스트 코드
    try:
        from src.data_loader import load_or_download
        from src.config import DATA_DIR, DATA_START
    except ImportError:
        from data_loader import load_or_download
        from config import DATA_DIR, DATA_START
        
    print("=== 피처 엔지니어링 테스트 ===")
    symbol = "AAPL"
    
    try:
        df = load_or_download(symbol, DATA_DIR, DATA_START)
        print(f"원본 데이터: {df.shape}")
        
        df_features = add_basic_features(df)
        print(f"피처 추가 후: {df_features.shape}")
        print(f"추가된 컬럼: {[c for c in df_features.columns if c not in df.columns]}")
        
        X, y, dates, cols = make_tabular_dataset(df_features)
        print(f"\n테이블형 데이터셋 (XGBoost용):")
        print(f"X: {X.shape}, y: {y.shape}")
        
        X_seq, y_seq, dates_seq = make_supervised_dataset(df_features, window_size=20)
        print(f"\n시퀀스 데이터셋 (LSTM용):")
        print(f"X: {X_seq.shape}, y: {y_seq.shape}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
