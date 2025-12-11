"""
학습된 모델로부터 매매 신호를 생성하는 모델 예측 모듈입니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import joblib
import warnings

# 패키지 실행을 위한 절대 임포트
try:
    from src.features import add_basic_features
    from src.config import MODEL_DIR, DATA_DIR, DATA_START, DATA_END
    from src.data_loader import load_or_download
except ImportError:
    # 로컬 실행을 위한 상대 임포트 (fallback)
    from features import add_basic_features
    from config import MODEL_DIR, DATA_DIR, DATA_START, DATA_END
    from data_loader import load_or_download

warnings.filterwarnings('ignore')


def load_xgb_model(symbol: str) -> Tuple[object, object, list]:
    """
    학습된 XGBoost 모델, 스케일러, 피처 컬럼을 로드합니다.
    
    Args:
        symbol: 주식 티커 심볼
    
    Returns:
        model, scaler, feature_cols
    """
    model_path = MODEL_DIR / f"{symbol}_xgb_model.pkl"
    scaler_path = MODEL_DIR / f"{symbol}_xgb_scaler.pkl"
    features_path = MODEL_DIR / f"{symbol}_xgb_features.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_path}. 먼저 모델을 학습하세요.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    
    return model, scaler, feature_cols


def predict_probabilities_xgb(
    symbol: str,
    df: pd.DataFrame,
    model: object = None,
    scaler: object = None,
    feature_cols: list = None
) -> pd.Series:
    """
    XGBoost 모델을 사용하여 확률을 예측합니다.
    
    Args:
        symbol: 주식 티커 심볼
        df: OHLCV 데이터가 포함된 DataFrame (피처가 없으면 추가됨)
        model: 학습된 모델 (None이면 디스크에서 로드)
        scaler: 적합된 스케일러 (None이면 디스크에서 로드)
        feature_cols: 피처 컬럼 (None이면 디스크에서 로드)
    
    Returns:
        확률 시리즈 (입력 df와 인덱스 정렬됨)
    """
    # 모델이 제공되지 않은 경우 로드
    if model is None or scaler is None or feature_cols is None:
        model, scaler, feature_cols = load_xgb_model(symbol)
    
    # 피처가 아직 없으면 추가
    if 'return' not in df.columns:
        df = add_basic_features(df)
    
    # NaN 행 제거
    df_clean = df.dropna()
    
    if df_clean.empty:
        raise ValueError("NaN 제거 후 유효한 데이터가 없습니다.")
    
    # 피처 추출
    try:
        X = df_clean[feature_cols].values
    except KeyError as e:
        missing_cols = set(feature_cols) - set(df_clean.columns)
        raise ValueError(f"누락된 피처 컬럼: {missing_cols}")
    
    # 피처 스케일링
    X_scaled = scaler.transform(X)
    
    # 확률 예측
    proba = model.predict_proba(X_scaled)[:, 1]  # 클래스 1(상승)의 확률
    
    # 원본 인덱스로 시리즈 생성
    proba_series = pd.Series(proba, index=df_clean.index, name='prob_up')
    
    return proba_series


def generate_signal_from_probs(
    probs: pd.Series,
    long_threshold: float = 0.55,
    flat_threshold: float = 0.45
) -> pd.Series:
    """
    확률 예측으로부터 매매 신호를 생성합니다.
    
    Args:
        probs: 확률 시리즈 (0-1)
        long_threshold: 매수 신호 임계값
        flat_threshold: 매도/관망 신호 임계값
    
    Returns:
        신호 시리즈: 1 (매수), 0 (관망), -1 (매도, 현재 사용 안 함)
    """
    signals = pd.Series(0, index=probs.index, name='signal')
    
    # 매수 신호: 상승 확률이 높음
    signals[probs >= long_threshold] = 1
    
    # 관망 신호: 확률이 낮거나 불확실함
    signals[probs < flat_threshold] = 0
    
    # 향후 확장: 매도 신호 추가 가능
    # signals[probs < short_threshold] = -1
    
    return signals


def predict_and_generate_signals(
    symbol: str,
    df: pd.DataFrame = None,
    long_threshold: float = 0.55,
    flat_threshold: float = 0.45,
    days: Optional[int] = None
) -> pd.DataFrame:
    """
    전체 예측 파이프라인: 데이터 로드, 예측, 신호 생성.
    
    Args:
        symbol: 주식 티커 심볼
        df: OHLCV 데이터가 포함된 DataFrame. None이면 캐시에서 로드.
        long_threshold: 매수 신호 임계값
        flat_threshold: 관망 신호 임계값
        days: 지정된 경우 최근 N일만 사용
    
    Returns:
        DataFrame 컬럼: close, prob_up, signal
    """
    # 데이터가 제공되지 않은 경우 로드
    if df is None:
        df = load_or_download(symbol, DATA_DIR, DATA_START, DATA_END)
    
    # 지정된 경우 최근 N일만 사용
    if days is not None:
        df = df.tail(days)
    
    # 확률 예측
    probs = predict_probabilities_xgb(symbol, df)
    
    # 신호 생성
    signals = generate_signal_from_probs(probs, long_threshold, flat_threshold)
    
    # 결과 DataFrame 결합
    result = pd.DataFrame({
        'close': df.loc[probs.index, 'close'],
        'prob_up': probs,
        'signal': signals
    })
    
    return result


def get_latest_signal(symbol: str, days: int = 60) -> dict:
    """
    심볼에 대한 가장 최근 매매 신호를 가져옵니다.
    
    Args:
        symbol: 주식 티커 심볼
        days: 분석할 최근 일수
    
    Returns:
        최신 신호 정보가 담긴 딕셔너리
    """
    result = predict_and_generate_signals(symbol, days=days)
    
    latest = result.iloc[-1]
    
    signal_info = {
        'symbol': symbol,
        'date': result.index[-1],
        'close': latest['close'],
        'probability': latest['prob_up'],
        'signal': int(latest['signal']),
        'signal_name': {1: 'LONG', 0: 'FLAT', -1: 'SHORT'}[int(latest['signal'])],
        'recommendation': 'BUY' if latest['signal'] == 1 else ('SELL' if latest['signal'] == -1 else 'HOLD')
    }
    
    return signal_info


def batch_predict(symbols: list, days: int = 60) -> pd.DataFrame:
    """
    여러 심볼에 대한 신호를 예측합니다.
    
    Args:
        symbols: 주식 티커 심볼 리스트
        days: 분석할 최근 일수
    
    Returns:
        모든 심볼에 대한 예측 결과가 담긴 DataFrame
    """
    results = []
    
    for symbol in symbols:
        try:
            signal_info = get_latest_signal(symbol, days)
            results.append(signal_info)
            print(f"✓ {symbol}: {signal_info['signal_name']} (확률: {signal_info['probability']:.3f})")
        except Exception as e:
            print(f"✗ {symbol}: 오류 - {str(e)}")
            results.append({
                'symbol': symbol,
                'date': None,
                'close': None,
                'probability': None,
                'signal': None,
                'signal_name': 'ERROR',
                'recommendation': 'N/A'
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== 모델 예측 테스트 ===\n")
    
    symbol = "AAPL"
    
    # 모델 존재 여부 확인
    model_path = MODEL_DIR / f"{symbol}_xgb_model.pkl"
    
    if not model_path.exists():
        print(f"{symbol}에 대한 모델을 찾을 수 없습니다. 먼저 모델을 학습하세요:")
        print(f"  python src/model_train.py")
    else:
        print(f"{symbol} 모델 로드 중...")
        
        # 최근 60일에 대한 예측 생성
        print("\n최근 60일에 대한 예측 생성 중...")
        result = predict_and_generate_signals(symbol, days=60)
        
        print(f"\n예측 결과 형태: {result.shape}")
        print(f"\n마지막 10개 예측:")
        print(result.tail(10))
        
        # 신호 분포
        print(f"\n신호 분포:")
        print(result['signal'].value_counts().sort_index())
        
        # 최신 신호
        print(f"\n{'='*60}")
        print("최신 신호")
        print(f"{'='*60}")
        latest = get_latest_signal(symbol, days=60)
        for key, value in latest.items():
            print(f"{key}: {value}")
        
        # 배치 예측 테스트
        print(f"\n{'='*60}")
        print("배치 예측 테스트")
        print(f"{'='*60}\n")
        
        symbols = ["AAPL"]  # 학습된 모델이 있는 경우 더 많은 심볼 추가 가능
        batch_results = batch_predict(symbols, days=60)
        print("\n배치 결과:")
        print(batch_results.to_string(index=False))
